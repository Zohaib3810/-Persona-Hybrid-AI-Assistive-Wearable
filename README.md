```markdown
# Persona: Hybrid AI Assistive Wearable

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Ollama-black.svg)](https://ollama.ai/)
[![Vision Model](https://img.shields.io/badge/VLM-Moondream2-purple.svg)](https://github.com/vikhyat/moondream)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Persona** is an open-source, edge-driven assistive AI wearable architecture designed to assist visually impaired individuals. It integrates local Small Language Models (SLMs), offline wake-word recognition, and a cloud-assisted visual reasoning pipeline into a continuous sensory loop that translates the surrounding physical environment into real-time spoken audio.

---

##  Key Features

* **100% Offline Wake Word:** Edge detection powered by `openwakeword` and ONNX Runtime prevents constant audio streaming to the cloud and maintains complete user privacy.
* **Local SLM Reasoning:** Microsoft's **Phi-3** running on-device via **Ollama** executes general queries and conversational dialogue with zero latency and no internet connection.
* **Hybrid Vision Pipeline:** Automatic intent detection triggers OpenCV camera frame capture, sending visual queries to a remote **Moondream2** Vision-Language Model hosted on GPU infrastructure via an authenticated Ngrok tunnel.
* **Low-Latency Spoken Feedback:** Spoken system status and contextual descriptions are rendered instantaneously using the offline `pyttsx3` speech engine.
* **Intent Routing:** Segregates commands based on visual trigger keywords (e.g., *"what is"*, *"look at"*, *"read"*, *"front of"*) to optimize bandwidth and compute resources.

---

##  Architecture

```text
                     +---------------------------------------+
                     |        User Audio Environment         |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | OpenWakeWord Engine (Offline/ONNX)    |
                     +---------------------------------------+
                                         | (Trigger Activated)
                                         v
                     +---------------------------------------+
                     | Speech Recognition / Intent Parsing   |
                     +---------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
           (Text/General Query)                     (Visual Query Detected)
                     |                                       |
                     v                                       v
      +-----------------------------+         +-----------------------------+
      | Ollama Local Brain (Phi-3)  |         | OpenCV Frame Capture        |
      +-----------------------------+         +-----------------------------+
                     |                                       |
                     |                                       v
                     |                        +-----------------------------+
                     |                        | Remote Moondream2 API       |
                     |                        | (Google Colab via Ngrok)    |
                     |                        +-----------------------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | pyttsx3 Local Text-To-Speech (TTS)    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |             Spoken Audio              |
                     +---------------------------------------+

```

---

##  Project Structure

```bash
persona/
├── persona_local_prototype.py   # Client-side execution loop (Wake word, CV2, Ollama client, TTS)
├── persona_backend_colab.py     # Colab backend server (Moondream2, Flask API, PyNgrok tunnel)
├── requirements.txt             # Client dependencies
└── README.md                    # Project documentation

```

---

##  Hardware & Software Prerequisites

* **Hardware:** Webcam/Camera sensor, USB Microphone, Audio Output (Headphones/Speaker)
* **Local Environment:** Python 3.10+, [Ollama](https://ollama.ai/)
* **Remote Environment:** Google Colab (T4 GPU runtime) or any GPU-enabled cloud VM with an [Ngrok](https://ngrok.com/) account.

---

## 🚀 Setup & Installation

### 1. Local Client Configuration

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/persona.git](https://github.com/your-username/persona.git)
cd persona

```


2. **Create and activate a virtual environment:**
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```


3. **Install client-side dependencies:**
```bash
pip install opencv-python SpeechRecognition pyttsx3 requests openwakeword pyaudio onnxruntime numpy

```


4. **Initialize the local SLM:**
```bash
ollama pull phi3
ollama run phi3

```



---

### 2. Vision Backend Deployment (Google Colab)

1. Open Google Colab and set the runtime to **T4 GPU** (`Runtime > Change runtime type`).
2. Install the backend dependencies:
```bash
!pip install transformers accelerate einops flask pyngrok

```


3. Run the vision model server script:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import os

# Load Moondream2 VLM
model_id = "vikhyatk/moondream2"
revision = "2024-05-08"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision).to("cuda")

def analyze_visual_context(image_path, query):
    image = Image.open(image_path)
    enc_image = model.encode_image(image)
    return model.answer_question(enc_image, query, tokenizer)

# Configure Ngrok Tunnel
ngrok.kill()
PORT = 5050
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(PORT).public_url
print(f"API Online: {public_url}/analyze")

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files or 'query' not in request.form:
        return jsonify({'error': 'Missing image or query data'}), 400

    file = request.files['image']
    query = request.form['query']
    temp_path = "incoming_capture.jpg"
    file.save(temp_path)

    try:
        answer = analyze_visual_context(temp_path, query)
        os.remove(temp_path)
        return jsonify({'response': answer})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':PORT}).start()

```



---

### 3. Running Persona

1. Copy the generated Ngrok endpoint URL (e.g., `https://xxxx-xxxx.ngrok-free.dev/analyze`).
2. Update the `API_URL` variable in `persona_local_prototype.py`:
```python
API_URL = "[https://your-ngrok-url.ngrok-free.dev/analyze](https://your-ngrok-url.ngrok-free.dev/analyze)"

```


3. Execute the local client:
```bash
python persona_local_prototype.py

```


4. **Trigger Commands:**
* Say: `"Hey Mycroft"` (or your custom wake word).
* **General Query:** *"What time is it in Tokyo?"* (Handled locally by Phi-3)
* **Visual Query:** *"What is in front of me?"* / *"Read this text."* (Handled by Moondream2)
* **Shutdown:** *"Goodbye"* or *"Shut down"*



---

## Authors

* **Muhammad Zohaib**
* **Hamza Javed**

*BS Artificial Intelligence (BS AI - 6)*

---

##  License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
