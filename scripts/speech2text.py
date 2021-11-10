import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
args = parser.parse_args()

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio, rate = librosa.load(args.in_path, sr = 16000)

input_values = tokenizer(audio, return_tensors = "pt").input_values
logits = model(input_values).logits
prediction = torch.argmax(logits, dim = -1)
transcription = tokenizer.batch_decode(prediction)[0]

print(transcription)