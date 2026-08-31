import cv2
import speech_recognition as sr
import pyttsx3
import time

class LocalPersonaInterface:
    def __init__(self):
        # Initialize local text-to-speech engine for system status
        self.tts_engine = pyttsx3.init()
        # Set property for slightly faster speech for better UX
        self.tts_engine.setProperty('rate', 180) 
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Simple keywords to trigger the camera
        self.visual_triggers = ["what is", "look at", "front of", "read", "this", "see", "camera", "object", "picture"]

    def speak(self, text):
        """Provide instant spoken feedback to the user."""
        print(f"Persona: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_for_command(self):
        """Capture user voice and convert to text."""
        with sr.Microphone() as source:
            self.speak("I am listening.")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                self.speak("Processing your request...")
                
                # Using Google's free API for rapid prototyping.
                # In final build (all-local or all-cloud), you would swap
                # this with an offline model (Whisper).
                query = self.recognizer.recognize_google(audio).lower()
                print(f"User asked: {query}")
                return query
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything.")
                return None
            except sr.UnknownValueError:
                self.speak("I couldn't understand that.")
                return None

    def capture_image(self):
        """Snap a photo using the default camera and save locally."""
        self.speak("Taking a look now.")
        
        # 0 is usually the default camera. Swap to 1 for external USB.
        cap = cv2.VideoCapture(0) 
        
        if not cap.isOpened():
            self.speak("Error accessing the camera.")
            return None

        # Give the camera 1 second to adjust to lighting
        time.sleep(1) 
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Save the image for Phase 2 processing
            file_name = "current_capture.jpg"
            cv2.imwrite(file_name, frame)
            self.speak("Image captured successfully.")
            return file_name
        else:
            self.speak("Failed to capture image.")
            return None

    def analyze_intent(self, query):
        """Check if the user's question requires visual context."""
        if any(trigger in query for trigger in self.visual_triggers):
            return True
        return False
    
    def run(self):
        """Production loop using OpenWakeWord (100% Free & Offline)."""
        import requests 
        import time
        import pyaudio
        import numpy as np
        from openwakeword.model import Model
        import openwakeword

        # ---> MAKE SURE YOUR CURRENT NGROK URL IS HERE <---
        API_URL = "https://payphone-supper-coleslaw.ngrok-free.dev/analyze" 
        
        self.speak("Loading offline wake word engine...")
        
        # ---> ADD THIS LINE TO DOWNLOAD THE MISSING MODELS <---
        openwakeword.utils.download_models()
       # We are adding the inference_framework command to force it to use ONNX
        oww_model = Model(wakeword_models=["hey_mycroft"], inference_framework="onnx")
        
        # 2. Setup the Microphone Stream
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280
        
        audio = pyaudio.PyAudio()
        mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        self.speak("System online. Say 'Hey Mycroft' to wake me up.")
        
        try:
            # The Infinite Wake Word Loop
            while True:
                # 3. Read a tiny chunk of audio
                audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                
                # 4. Feed it to the AI
                prediction = oww_model.predict(audio_data)
                
                # 5. Check if "Hey Mycroft" was spoken (Confidence > 0.5)
                for mdl in oww_model.prediction_buffer.keys():
                    if oww_model.prediction_buffer[mdl][-1] > 0.5:
                        
                        self.speak("Yes?")
                        
                        # Reset the wake word buffer so it doesn't trigger twice
                        oww_model.reset() 
                        
                        # --- 6. TRIGGER THE MAIN BRAIN ---
                        query = self.listen_for_command()
                        
                        if query:
                            # The Kill Switch
                            if "goodbye" in query or "shutdown" in query:
                                self.speak("Goodbye! Shutting down Persona.")
                                mic_stream.stop_stream()
                                mic_stream.close()
                                audio.terminate()
                                return # Exits the whole program
                                
                            # Vision Check
                            needs_vision = self.analyze_intent(query)
                            
                            if needs_vision:
                                image_path = self.capture_image()
                                if image_path:
                                    self.speak("Sending visual data to the brain.")
                                    try:
                                        with open(image_path, 'rb') as f:
                                            files = {'image': f}
                                            data = {'query': query}
                                            response = requests.post(API_URL, files=files, data=data)
                                            
                                        if response.status_code == 200:
                                            brain_answer = response.json().get('response')
                                            self.speak(brain_answer)
                                        else:
                                            self.speak("The vision brain encountered an error.")
                                    except Exception as e:
                                        print(f"Network Error: {e}")
                                        self.speak("I could not connect to the Colab Brain.")
                                        
                            # Standard Conversation (Ollama)
                            else:
                                self.speak("Thinking...")
                                try:
                                    ollama_url = "http://localhost:11434/api/generate"
                                    payload = {"model": "phi3", "prompt": query, "stream": False}
                                    response = requests.post(ollama_url, json=payload)
                                    
                                    if response.status_code == 200:
                                        text_answer = response.json().get('response')
                                        clean_answer = text_answer.replace("*", "").replace("#", "")
                                        self.speak(clean_answer)
                                    else:
                                        print(f"OLLAMA ERROR: {response.status_code} - {response.text}")
                                        self.speak("My local text brain returned an error.")
                                except requests.exceptions.RequestException:
                                    self.speak("I couldn't reach Ollama.")
                        
                        # After answering, it goes quiet again.
                        print("Listening for wake word again...")
                        
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Shutting down...")
        finally:
            mic_stream.stop_stream()
            mic_stream.close()
            audio.terminate()

# Run the assistant prototype
if __name__ == "__main__":
    assistant = LocalPersonaInterface()
    assistant.run()