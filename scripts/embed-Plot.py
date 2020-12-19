import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import MDS
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import MDS
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
import argparse
import random
import time 

#############################################################################################
parser = argparse.ArgumentParser()

# IO parameters
parser.add_argument('--input_dir', type=str)
parser.add_argument('--export_dir', type=str)
# model parameters
parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--num_iter', type=int,default=300)
parser.add_argument('--num_class',type=int,default=13)

args = parser.parse_args()
#############################################################################################

class nlu_dataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __getitem__(self, index):
        
        text = str(self.data.UTTERANCE[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'intent' : self.data.INTENT[index],
        'lang' : self.data.LANGUAGE[index]}

    def __len__(self):
        return len(self.data)


print("*"*10,"loading encoder","*"*10)
encoder = DistilBertModel.from_pretrained(args.model)
print("*"*10,"loading data","*"*10)
data = pd.read_csv(args.input_dir,sep='\t')
print("*"*10,"loading tokenizer","*"*10)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# fetching representation of sentences from encoder
print("*"*10,"creating dataloader","*"*10)
trainDS = nlu_dataset(args.input_dir,tokenizer,46)
trainDL = DataLoader(trainDS,32)

print("*"*10,"encoding sentences into vectors","*"*10)
reps = []
for idx,batch in enumerate(trainDL,0):
    ids,mask,intent,lang = batch['ids'],batch['mask'],batch['intent'],batch['lang']
    hidden = encoder(input_ids=ids, attention_mask=mask)[0][:,0]
    reps.append(hidden.detach())

Rep = torch.cat(reps).type(torch.DoubleTensor)
print("*"*10,"MDS algorithm on hidden representation","*"*10)

# reducing dimension using MDS algorithm
embedding = MDS(n_components=2, max_iter=args.num_iter, metric=True, n_init=10)
reduced_Rep = embedding.fit_transform(Rep)

data['Dim1'] = Rep[:,0] 
data['Dim2'] = Rep[:,1]

# generating plot and saving it to file
print("*"*10,"plotting and saving the figure!","*"*10)
plt.figure(figsize=(14,10))

color = sns.color_palette("hls",args.num_class)

sns.scatterplot(x='Dim1', y='Dim2', data=data ,palette=color,
                hue='INTENT', style='LANGUAGE', alpha=1)

plt.savefig(args.export_dir)