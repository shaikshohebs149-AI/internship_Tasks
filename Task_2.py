import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_wav2vec(audio_path):
    # 1. Load pre-trained model and processor
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # 2. Load and preprocess the audio file (must be 16kHz)
    speech, rate = librosa.load(audio_path, sr=16000)
    
    # 3. Tokenize and Predict
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits

    # 4. Decode the IDs to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Usage: 
# print(transcribe_wav2vec("path_to_your_audio.wav"))