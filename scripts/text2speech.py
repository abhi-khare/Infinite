import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import argparse

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
parser.add_argument('--out_path', type=str)
args = parser.parse_args()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# loading data
data = pd.read_csv(args.in_path,sep='\t')

# loading models
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

# rearranging rows of dataframe according to sequence length
s = data.TEXT.str.len().sort_values(ascending=False).index
sorted_data = data.reindex(s)
sorted_data =  sorted_data.reset_index(drop=True)

sorted_sequence = list(sorted_data['TEXT'])
chunked_sequence = chunks(sorted_sequence,32)

cnt = 0

for idx,chunk in enumerate(list(chunked_sequence)):
    
    print('processing {} chunk'.format(idx))
    
    with torch.inference_mode():
        processed, lengths = processor(chunk)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)
    
    for jdx in range(len(chunk)):
        torchaudio.save(args.out_path + "/{}.wav".format(int(cnt+idx)), waveforms[idx : (idx+1)].cpu(), sample_rate=vocoder.sample_rate)
    cnt += len(chunk)
