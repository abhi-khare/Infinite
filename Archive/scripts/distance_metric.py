from sklearn.manifold import MDS
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import argparse
import random
from prettytable import PrettyTable
import time 
import numpy

parser = argparse.ArgumentParser()

# IO parameters
parser.add_argument('--input_dir', type=str)
parser.add_argument('--export_dir', type=str)
# model parameters
parser.add_argument('--model', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--num_class',type=int,default=13)

args = parser.parse_args()

def euclid_dist(rep1,rep2):
    return torch.sum(torch.abs(rep1-rep2)**2)**0.5


class evaluateDS(Dataset):
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
trainDS = evaluateDS(args.input_dir,tokenizer,46)
trainDL = DataLoader(trainDS,32)

print("*"*10,"encoding sentences into vectors","*"*10)
reps = []
for idx,batch in enumerate(trainDL,0):
    ids,mask,intent,lang = batch['ids'],batch['mask'],batch['intent'],batch['lang']
    hidden = encoder(input_ids=ids, attention_mask=mask)[0][:,0]
    reps.append(hidden.detach())

Rep = torch.cat(reps).type(torch.DoubleTensor)
Intent = list(data.INTENT)

data_dict = {key:[] for key in list(data.INTENT.unique())}

for i in range(len(Rep)):
    data_dict[Intent[i]].append(Rep[i,:])

#calculating inter-class distance

class_center = []

for key,rep in data_dict.items():
    center,cnt = 0,0
    for i in range(len(rep)):
        center += rep[i]
        cnt += 1
    class_center.append(center/cnt)

#print(len(data_dict.keys()))
inter_class = []

for i in range(len(class_center)):
    perClsDist = []
    for j in range(len(class_center)):
        perClsDist.append(euclid_dist(class_center[i],class_center[j]))
    
    inter_class.append(perClsDist)

interTable = PrettyTable()
interTable.field_names =  ["$"] + list(data_dict.keys())


for idx, key in enumerate(data_dict.keys()):
    row = [key]
    for j in range(len(inter_class[idx])):
        #print(inter_class[idx][j])
        row.append(inter_class[idx][j].data.cpu().numpy())
    #print(len(row))
    interTable.add_row(row)
interTableStr = interTable.get_string()

text_file = open(args.export_dir + "interClassDist.txt", "w")
n = text_file.write(interTableStr)
text_file.close()

#calculating intra-class distance

intra_dist = []
for key,rep in data_dict.items():
    total_dist,cnt = 0,0
    for i in range(len(rep)-1):
        for j in range(i+1,len(rep)):
            total_dist+=euclid_dist(rep[i],rep[j])
            cnt += 1
    intra_dist.append(total_dist/float(cnt))

intraTable = PrettyTable()

intraTable.field_names = ["Intent","Intra Class Distance"]
for idx,key in enumerate(list(data_dict.keys())):
    intraTable.add_row([key,intra_dist[idx]])


print(intraTable)

intraTableStr = intraTable.get_string()

text_file = open(args.export_dir + "intraClassDist.txt", "w")
n = text_file.write(intraTableStr)
text_file.close()
    
