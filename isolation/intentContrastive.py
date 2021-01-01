import time,os 
import torch 
import torch.nn as nn
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses
from dataset import nluDataset
from torch.utils.tensorboard import SummaryWriter
from utils import getSlotsLabels
from pytorch_metric_learning import samplers
import pandas as pd
import argparse


# params
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--num_worker', type=int , default=4)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--model_export', type=str)
parser.add_argument('--num_iter',type=int,default=250)
parser.add_argument('--max_len',type=int,default=56)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--lr',type=int,default=0.00001)
parser.add_argument('--weights',type=int,default=0.0001)
parser.add_argument('--exp_name',type=str)

parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

# data
parser.add_argument('--intent_num', type=int, default=17)
parser.add_argument('--slots_num', type=int , default=160)


args = parser.parse_args()

writer = SummaryWriter('logs/ICPT/'+args.exp_name)


class Bertencoder(nn.Module):

    def __init__(self,model):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 256)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden

model = Bertencoder(args.model_name).to(device=args.device)

trainDS, valDS = nluDataset(args.train_dir,args.tokenizer_name,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_name,args.max_len,args.device)

train_labels = list(pd.read_csv('./data/train_EN.tsv',sep='\t').intent_ID)
val_labels = list(pd.read_csv('./data/dev_EN.tsv',sep='\t').intent_ID)

sampler_train = samplers.MPerClassSampler(train_labels, 6, batch_size=None, length_before_new_iter=30000)
sampler_val = samplers.MPerClassSampler(val_labels, 3, batch_size=None, length_before_new_iter=5000)

trainDL = DataLoader(trainDS,batch_size=args.intent_num*6,num_workers=args.num_worker, sampler=sampler_train)
valDL = DataLoader(valDS,batch_size=args.intent_num*6,num_workers=args.num_worker,sampler=sampler_val)

optimizer =  optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weights)

# experiment 1 triplet margin loss margin 2.0
loss_func = losses.TripletMarginLoss(margin=2.0)

def validation(model,val_DL,epoch):
    model.eval()
    val_loss ,num_batch = 0.0,0
    with torch.no_grad():
        for batch in valDL:
                
            num_batch +=1
            text_ids = batch['token_ids'].to(args.device, dtype = torch.long)
            text_mask = batch['mask'].to(args.device, dtype = torch.long)
            labels = batch['intent_id'].to(args.device, dtype = torch.long)

            embeddings = model(text_ids,text_mask)
            contraLoss = loss_func(embeddings,labels)
            val_loss += contraLoss.detach()

    val_loss = val_loss/float(num_batch)

    return val_loss  

# training loop
num_iter = 0
best_loss = 1000.0
for train_batch in trainDL:

    model.train()
    
    num_iter += 1
    
    text_ids = train_batch['token_ids'].to(args.device, dtype = torch.long)
    text_mask = train_batch['mask'].to(args.device, dtype = torch.long)
    label = train_batch['intent_id'].to(args.device, dtype = torch.long)

    embeddings = model(text_ids,text_mask)
    
    optimizer.zero_grad()
    contraLoss = loss_func(embeddings,label)
    contraLoss.backward()
    optimizer.step()

    if num_iter % 20 == 0:
        val_loss = validation(model,valDL,num_iter)

        print('iter:',num_iter,'val_loss:', val_loss)
        if best_loss > val_loss:
            best_loss = val_loss
            model.encoder.save_pretrained(args.model_export)
