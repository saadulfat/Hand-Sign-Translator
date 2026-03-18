# Unified Agentic Sign Language Translator

**A cutting-edge solution for interpreting Sign Language sequences from both Images and Videos, transforming them into context-aware, emotionally intelligent sentences, and speaking them aloud.**

This project uses an **Agentic Flow** architecture combined with Google's powerful **Gemini 2.5 Flash** language model to intelligently string together detected sign language gestures into natural, human-like sentences.

---

## 🚀 Key Features

- **Agentic Workflows**: Clean segregation of responsibilities:
  - **VisionAgent**: Locates hands via *MediaPipe Hands* and predicts the exact sign via a custom-trained **Multilayer Perceptron (MLP)** model. The model processes 126 key hand landmarks to classify static gestures dynamically.
  - **LanguageAgent**: Takes the raw sequence of words and feeds them to *Gemini 1.5 Flash* with a powerful, specialized prompt to construct grammatically sound, emotionally intelligent English sentences.
  - **SpeechAgent**: Rapidly converts the output sentence into English Audio (via `gTTS`) and Chinese Audio (via `gTTS` / `googletrans`).
- **High-Performance Video Parsing**: Capable of processing video files efficiently by intelligently skipping frames (processing ~10 frames per second) without losing sequence integrity or bounding box accuracy.
- **Unified UI**: An intuitive Gradio interface combining both **Images Flow** (for sequences of photos) and **Video Flow** into one seamless application.

---

## 🧠 Model Architecture & Technologies

### 1. Vision and Landmark Detection (MediaPipe)
We use **Google MediaPipe Hands** to infer 21 3D landmarks of a hand from just a single frame. Up to 2 hands are detected per frame, resulting in 42 individual (x, y, z) coordinates, which are flattened into a **126-dimensional feature vector** (padded with negative values if only one hand is detected).

### 2. Sign Language Classification (TensorFlow/Keras MLP)
The core gesture classification is powered by a robust **Multilayer Perceptron (MLP)** built using Keras/TensorFlow. 
- **Input:** 126 numerical landmark coordinates.
- **Hidden Layers:** A fully connected dense neural network, trained explicitly on diverse combinations of sign language gesture variations.
- **Output:** A softmax classification across our target vocabulary of signs.

### 3. Natural Language Generation (Gemini 2.5 Flash)
Translating raw signs (e.g., `["Me", "Drink", "Water"]`) into natural language requires grammatical smoothing. We leverage **Google Gemini 2.5 Flash** via an agentic prompt designed specifically to fix inflections and provide an emotionally intelligent, brief English sentence (e.g., *"I would like to get a drink of water."*).

---

## 🛠 Prerequisites & Installation

### 1. Python Environment

Ensure you have **Python 3.10+** installed. It is strongly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv signlang-env

# Activate the virtual environment
# On Windows:
signlang-env\Scripts\activate
# On Mac/Linux:
source signlang-env/bin/activate
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Model Files

Ensure the following pre-trained models are located in the root directory:
- `Model (3).h5`: The compiled TensorFlow/Keras MLP model.
- `Model (3).pkl`: The Label Encoder to map predictions to human-readable string labels.

### 4. Environment Variables

Create a file named `config.env` in the root directory and add your Google Gemini API key:

```env
GEMINI_API_KEY="your_api_key_here"
```

---

## 🏃‍♂ How to Run

To launch the web interface locally, run:

```bash
python hand_sign.py
```

- A local URL (e.g., `http://127.0.0.1:8080`) will be printed in the console.
- Open this URL in your web browser.
- **Public Share File**: A public `*.gradio.live` link will also be generated automatically so that you can easily share the application online.

---

## 📸 Usage Guide

### Images Flow Tab
1. Select the **Images Flow** tab.
2. Upload a sequence of images containing hand gestures in chronologic order.
3. Click **"Translate Images"**.
4. The system will detect the signs in each image, translate the aggregate into a meaningful sentence, and generate audio for it.

### Video Flow Tab
1. Select the **Video Flow** tab.
2. Upload a short video file (`.mp4`, `.mov`, etc.) where someone is gesturing in sign language.
3. Click **"Process Video"**.
4. The Vision Agent processes the video (with high-speed frame skipping), draws the landmarks onto the video, while the Language and Speech Agents construct the sentence and generated translations simultaneously.

---

## 🔧 Troubleshooting

- **No GEMINI_API_KEY provided:** The application will fall back to simply connecting the raw predicted words with spacing.
- **Missing Audio Translations:** Try restarting the server if gTTS or Google Translate API timeouts occur.
- **Video takes too long:** The application automatically attempts to process every Nth frame (targeting ~10 fps) to dramatically increase throughput on lengthy videos. If you need it even faster, increase the `skip_frames` constant within `agentic_app.py`.

---
*Built with ❤️ using Gradio, MediaPipe, TensorFlow, and Google Gemini.*
