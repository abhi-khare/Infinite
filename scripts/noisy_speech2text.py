import torch 
import librosa
import pandas as pd
from os import listdir
from os.path import isfile, join
from torch_audiomentations import Compose, AddBackgroundNoise
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise_data_path', type=str , help="path to directory that contains all the background noise audio samples")
parser.add_argument('--sample_path', type=str , help= "path to directory that contains clean audio samples")
parser.add_argument('--out_path', type=str , help="path to save the noisy text dataset")
parser.add_argument('--dataset_path', type=str , help="path to testset which has been injected with noise")
args = parser.parse_args()


# setting up compute device
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loading STT models
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/efs-storage/research/preTrained_models/ASR/tokenizer/")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir= "/efs-storage/research/preTrained_models/ASR/model/")

# composing augmentation
apply_augmentations = Compose(transforms=[AddBackgroundNoise(
                      background_paths=args.noise_data_path,
                      p=1.0,
                      min_snr_in_db=-2.5,
                      max_snr_in_db=2.5,
                      )])

# loading clean dataset
original_data = pd.read_csv(args.dataset_path,sep='\t')
original_data = original_data[['ID','INTENT','INTENT_ID']]

# collecting audio sample paths
print(args.sample_path)

sample_path = [ args.sample_path + f for f in listdir(args.sample_path) if isfile(join(args.sample_path, f))]

# converting text 2 speech
for idx in range(5):

    print('processing version {}'.format(idx))

    noisy_samples = []

    for jdx,path in enumerate(sample_path):
        print(path)
        audio, rate = librosa.load(path, sr = 16000)
        audio = (torch.from_numpy(audio))
        new_audio = torch.unsqueeze(audio,0)
        new_audio = torch.unsqueeze(new_audio,0)
        perturbed_audio_samples = apply_augmentations(new_audio, sample_rate=16000)
        perturbed_audio_samples = torch.squeeze(perturbed_audio_samples,0)
        perturbed_audio_samples = torch.squeeze(perturbed_audio_samples,0)

        input_values = tokenizer(perturbed_audio_samples, return_tensors = "pt").input_values
        logits = model(input_values).logits
        prediction = torch.argmax(logits, dim = -1)
        transcription = tokenizer.batch_decode(prediction)[0]
        noisy_samples.append(transcription)
    
    original_data['TEXT'] = noisy_samples
    original_data = original_data[["ID","TEXT",'INTENT',"INTENT_ID"]]

    original_data.to_csv( args.out_path + 'test_{}.csv'.format(idx),sep='\t', index=False)