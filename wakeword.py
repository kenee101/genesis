from faster_whisper import WhisperModel
from llama_cpp import Llama
from bark import generate_audio, preload_models
from scipy.io.wavfile import write
import sounddevice as sd
from playsound import playsound
from pvporcupine import Porcupine
import pvporcupine
import struct
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

# --- SETUP ---
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
WAKE_WORD = "hey genesis"
MODEL_PATH = "./llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

SAMPLE_RATE = 16000
CHUNK = 512
asr_model = WhisperModel.load_model("base")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
preload_models()

# --- Functions ---
def speak(text):
    audio = generate_audio(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, rate=24000, data=audio)
        playsound(tmp.name)
        os.remove(tmp.name)

def record_phrase(seconds=5):
    print("🎤 Listening...")
    recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    filename = "spoken.wav"
    write(filename, SAMPLE_RATE, recording)
    return filename

def transcribe(file):
    result = asr_model.transcribe(file)
    return result['text']

def chat_with_llm(prompt, history=[]):
    full_prompt = ""
    for u, a in history:
        full_prompt += f"User: {u}\nAssistant: {a}\n"
    full_prompt += f"User: {prompt}\nAssistant:"
    result = llm(full_prompt, max_tokens=256, stop=["User:"], temperature=0.7)
    return result["choices"][0]["text"].strip()

# --- Wake Word Detection + Chat Loop ---
chat_history = []

print("🎧 Waiting for 'Hey Genesis'...")

porcupine = Porcupine(
    access_key=ACCESS_KEY,
    keyword_paths=[pvporcupine.KEYWORD_PATHS[WAKE_WORD]],
)

audio_stream = sd.RawInputStream(samplerate=porcupine.sample_rate,
                                 blocksize=porcupine.frame_length,
                                 dtype='int16',
                                 channels=1)
audio_stream.start()

try:
    while True:
        pcm = audio_stream.read(porcupine.frame_length)[0]
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        result = porcupine.process(pcm)

        if result >= 0:
            print("🟢 Wake word detected! Listening for command...")
            filepath = record_phrase()
            user_input = transcribe(filepath)
            print("You said:", user_input)

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Exiting...")
                break

            response = chat_with_llm(user_input, chat_history)
            print("🤖", response)
            speak(response)
            chat_history.append((user_input, response))

except KeyboardInterrupt:
    print("🔴 Stopped.")
finally:
    audio_stream.stop()
    porcupine.delete()