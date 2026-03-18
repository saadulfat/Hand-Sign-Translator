import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import joblib
import google.generativeai as genai
import re
import asyncio
from gtts import gTTS
from googletrans import Translator
import gradio as gr
import tempfile
import shutil
import warnings
import threading
from collections import Counter
from typing import List, Tuple, Optional, Any

try:
    import edge_tts  # type: ignore
except Exception:
    edge_tts = None
    print("⚠️ edge_tts not installed; Chinese audio generation will be skipped.")

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------
# Config loader (simple .env-style parser)
# ---------------------------------------------------------------------
def _load_env_file(env_path: str) -> None:
    try:
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return

_load_env_file(os.path.join(os.path.dirname(__file__), "config.env"))


# ---------------------------------------------------------------------
# Base setup paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_H5_PATH = os.path.join(BASE_DIR, "Model (3).h5")
MODEL_PKL_PATH = os.path.join(BASE_DIR, "Model (3).pkl")


# ---------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------
class VisionAgent:
    """Handles extracting hand landmarks from images/video frames and predicting sign class."""
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self._initialized = False
        self._lock = threading.Lock()

    def _normalize_label(self, label: str) -> str:
        aliases = {
            "Early": "Drink",
            "School": "Water"  # Fixing misclassification where Water is detected as School
        }
        return aliases.get(label, label)
        
    def setup(self) -> Tuple[bool, str]:
        try:
            with self._lock:
                if self._initialized:
                    return True, "Already initialized"
                if not os.path.exists(MODEL_H5_PATH):
                    return False, f"Model file not found: {MODEL_H5_PATH}"
                if not os.path.exists(MODEL_PKL_PATH):
                    return False, f"Encoder file not found: {MODEL_PKL_PATH}"
                
                self.model = load_model(MODEL_H5_PATH, compile=False)
                self.label_encoder = joblib.load(MODEL_PKL_PATH)
                
                mp_hands = mp.solutions.hands
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self._initialized = True
                return True, "Models loaded successfully"
        except Exception as e:
            return False, f"Error initializing VisionAgent: {e}"

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = False) -> Tuple[Optional[str], float, np.ndarray]:
        if draw_landmarks:
            output_frame = frame.copy()
        else:
            output_frame = frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        
        predicted_class = None
        confidence = 0.0
        
        if result.multi_hand_landmarks:
            keypoints = []
            
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                if i >= 2: break
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(output_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            
            # If only 1 hand, pad the remaining 63 keypoints
            if len(result.multi_hand_landmarks) == 1:
                keypoints.extend([-1.0] * 63)
            
            # Pad or truncate just in case
            while len(keypoints) < 126: keypoints.extend([0.0, 0.0, 0.0])
            keypoints = keypoints[:126]
            
            if len(keypoints) == 126:
                X_input = np.array(keypoints).reshape(1, -1)
                try:
                    prediction = self.model.predict(X_input, verbose=0)
                    predicted_index = np.argmax(prediction)
                    raw_label = self.label_encoder.inverse_transform([predicted_index])[0]
                    predicted_class = self._normalize_label(raw_label)
                    confidence = float(np.max(prediction))
                except Exception as e:
                    print(f"Prediction error: {e}")
                    
        return predicted_class, confidence, output_frame

    def predict_image(self, image_path: str) -> Optional[Tuple[str, float]]:
        if not os.path.exists(image_path): return None
        frame = cv2.imread(image_path)
        if frame is None: return None
        
        h, w = frame.shape[:2]
        if w > 1024:
            frame = cv2.resize(frame, (int(w * 1024/w), int(h * 1024/w)))
            
        cls, conf, _ = self.process_frame(frame, draw_landmarks=False)
        return (cls, conf) if cls else None

    def predict_video(self, video_path: str, output_path: str, fps_override: int = 0) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        fps = fps_override if fps_override > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check orientation via quick read
        ret, probe_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        rotate_video = False
        if probe_frame is not None and probe_frame.shape[1] > probe_frame.shape[0]:
            # Standardizing portrait vs landscape if necessary. Here we just rotate for debugging.
            rotate_video = True
            width, height = height, width

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Performance Optimization: Process every Nth frame 
        # For a ~30FPS video, predicting every 3rd or 4th frame is usually good enough.
        skip_frames = max(1, int(fps / 10)) # Target around 10 FPS for predictions
        
        frame_interval = int(fps * 2) # check predictions over 2-second windows
        frame_count = 0
        prediction_window = []
        final_sequence = []
        
        # Variables to persist annotations across skipped frames
        last_cls = None
        last_conf = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if rotate_video:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
            # Only run the heavy MediaPipe + Keras prediction on a subset of frames
            if frame_count % skip_frames == 0:
                cls, conf, annotated_frame = self.process_frame(frame, draw_landmarks=True)
                last_cls = cls
                last_conf = conf
                
                if cls and conf >= 0.8:
                    prediction_window.append(cls)
            else:
                annotated_frame = frame.copy()
            # Always draw the ongoing prediction on the frame
            # Placing it in a highly visible semi-transparent background box for readability
            if last_cls and last_conf >= 0.7:
                text = f"Sign: {last_cls} | Acc: {last_conf*100:.1f}%"
                
                # Draw a background rectangle for better text visibility
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                cv2.rectangle(annotated_frame, (15, 20), (15 + tw + 20, 20 + th + 20), (0, 0, 0), -1)
                cv2.putText(annotated_frame, text, (25, 20 + th + 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % frame_interval == 0 and prediction_window:
                most_common = Counter(prediction_window).most_common(1)[0][0]
                if not final_sequence or final_sequence[-1] != most_common:
                    final_sequence.append(most_common)
                prediction_window = []
                
        cap.release()
        out.release()
        return final_sequence


# ---------------------------------------------------------------------
# Language Agent
# ---------------------------------------------------------------------
class LanguageAgent:
    """Uses Gemini 2.5 to form well-constructed sentences."""
    def __init__(self):
        self.model = None
        self._setup_gemini()
        
    def _setup_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ Missing GEMINI_API_KEY environment variable. Language Agent will fallback to basic joining.")
            return
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_sentence(self, words_list: List[str]) -> str:
        if not words_list:
            return "No signs detected."
            
        # Deduplicate continuous words
        deduped = []
        for w in words_list:
            if not deduped or deduped[-1] != w:
                deduped.append(w)
                
        words_str = ", ".join(deduped)
        
        if not self.model:
            return f"{' '.join(deduped)}."
            
        prompt = (
            "You are an expert Sign Language-to-English interpreter. "
            "Convert the sequence of sign language keywords into ONE grammatically correct, emotionally intelligent English sentence.\n"
            f"Keywords: {words_str}\n"
            "\nRules:\n"
            "- Output exactly ONE short sentence (6-12 words).\n"
            "- Fix inflections (e.g., drink -> to drink / drinking).\n"
            "- You may add connecting words like I, me, want, please, need, the, to, a.\n"
            "- Keep the core meaning of the signs.\n"
            "- Do not output lists or raw keywords.\n"
            "Sentence:"
        )
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            sentences = re.findall(r'[^.!?]*[.!?]', text)
            sentence = sentences[0].strip() if sentences else text
            
            # Simple quality gate
            if len(sentence.split()) < 3:
                sentence = f"I want to {', '.join(deduped[:4]).lower()}."
            return sentence
        except Exception as e:
            print(f"Gemini API error: {e}")
            return f"{' '.join(deduped)}."


# ---------------------------------------------------------------------
# Speech Agent
# ---------------------------------------------------------------------
class SpeechAgent:
    """Handles text-to-speech generation for English and Chinese."""
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        
    async def generate_audio_async(self, sentence: str) -> Tuple[Optional[str], Optional[str]]:
        if not sentence or len(sentence.strip()) < 2: return None, None
        
        audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        import hashlib
        h = hashlib.md5(sentence.encode()).hexdigest()[:8]
        en_path = os.path.join(audio_dir, f"en_{h}.mp3")
        zh_path = os.path.join(audio_dir, f"zh_{h}.mp3")
        
        # English
        try:
            tts = gTTS(text=sentence, lang="en", slow=False)
            tts.save(en_path)
        except Exception as e:
            print(f"EN Audio Error: {e}")
            en_path = None
            
        # Chinese
        try:
            translator = Translator()
            translation = translator.translate(sentence, src="en", dest="zh-cn")
            chinese_text = translation.text
            
            # Using gTTS for Chinese as well to avoid edge_tts 403 Forbidden errors
            tts_zh = gTTS(text=chinese_text, lang="zh-CN", slow=False)
            tts_zh.save(zh_path)
            
            # For async compatibility, we use asyncio.sleep(0) to yield control
            await asyncio.sleep(0)
        except Exception as e:
            print(f"ZH Audio Error: {e}")
            zh_path = None
            
        return en_path, zh_path


# ---------------------------------------------------------------------
# Orchestrator Workflow
# ---------------------------------------------------------------------
class SignLanguageApp:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vision = VisionAgent()
        self.language = LanguageAgent()
        self.speech = SpeechAgent(self.temp_dir)
        
    def initialize(self) -> str:
        success, msg = self.vision.setup()
        return msg if not success else "Initialized successfully"

    def orchestrate_images(self, files: List[Any]) -> Tuple[str, str, Optional[str], Optional[str]]:
        if not files: return "No files", "Please upload images.", None, None
        
        words = []
        details = []
        
        for i, file in enumerate(files):
            file_path = file.name if hasattr(file, 'name') else str(file)
            res = self.vision.predict_image(file_path)
            if res:
                cls, conf = res
                words.append(cls)
                details.append(f"Image {i+1}: {cls} ({conf:.2f})")
            else:
                details.append(f"Image {i+1}: No hand detected")
                
        if not words:
            return "Results:\n" + "\n".join(details), "No signs detected.", None, None
            
        sentence = self.language.generate_sentence(words)
        
        try:
            en_audio, zh_audio = asyncio.run(self.speech.generate_audio_async(sentence))
        except Exception:
            en_audio, zh_audio = None, None
            
        results_text = f"Detected: {', '.join(words)}\n\nDetails:\n" + "\n".join(details)
        return results_text, sentence, en_audio, zh_audio

    def orchestrate_video(self, video_file: Any) -> Tuple[Optional[str], str, str, Optional[str], Optional[str]]:
        if not video_file: return None, "No video", "Upload a video.", None, None
        
        vpath = video_file.name if hasattr(video_file, 'name') else str(video_file)
        out_video = os.path.join(self.temp_dir, f"out_{os.path.basename(vpath)}.mp4")
        
        words = self.vision.predict_video(vpath, out_video)
        
        if not words:
            return out_video, "No signs", "No hands clearly detected.", None, None
            
        sentence = self.language.generate_sentence(words)
        
        try:
            en_audio, zh_audio = asyncio.run(self.speech.generate_audio_async(sentence))
        except Exception:
            en_audio, zh_audio = None, None
            
        return out_video, ", ".join(words), sentence, en_audio, zh_audio


# ---------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------
def build_interface():
    app_controller = SignLanguageApp()
    
    # Initialize proactively
    msg = app_controller.initialize()
    if "Error" in msg:
        print(f"Warning: {msg}")

    css = """
    .header { text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    """
    
    with gr.Blocks(css=css, title="Unified Agentic Sign Language Translator") as demo:
        gr.HTML("<div class='header'><h1>Sign Language to Speech Translator</h1><p>Powered by Agentic Flow</p></div>")
        
        with gr.Tabs():
            # ------------- Image Flow Tab -------------
            with gr.Tab("Images Flow"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload Sequences of Signs")
                        img_input = gr.File(file_count="multiple", file_types=["image"], label="Upload Images")
                        img_btn = gr.Button("Translate Images", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Results")
                        img_pred_text = gr.Textbox(label="Detected Signs & Details", lines=4)
                        img_sentence = gr.Textbox(label="Gemini Translated Sentence", lines=2)
                        with gr.Row():
                            img_audio_en = gr.Audio(label="English Audio")
                            img_audio_zh = gr.Audio(label="Chinese Audio")
                
                img_btn.click(
                    fn=app_controller.orchestrate_images,
                    inputs=[img_input],
                    outputs=[img_pred_text, img_sentence, img_audio_en, img_audio_zh]
                )

            # ------------- Video Flow Tab -------------
            with gr.Tab("Video Flow"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload Sign Language Video")
                        vid_input = gr.File(file_types=[".mp4", ".mov", ".avi"], label="Upload Video")
                        vid_btn = gr.Button("Process Video", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Results")
                        vid_out = gr.Video(label="Processed Video with Landmarks")
                        vid_pred_text = gr.Textbox(label="Detected Signs Sequence", lines=2)
                        vid_sentence = gr.Textbox(label="Gemini Translated Sentence", lines=2)
                        with gr.Row():
                            vid_audio_en = gr.Audio(label="English Audio")
                            vid_audio_zh = gr.Audio(label="Chinese Audio")
                            
                vid_btn.click(
                    fn=app_controller.orchestrate_video,
                    inputs=[vid_input],
                    outputs=[vid_out, vid_pred_text, vid_sentence, vid_audio_en, vid_audio_zh]
                )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=8080, inbrowser=True, share=True)
