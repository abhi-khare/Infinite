import time,os 
import torch 
import torch.nn as nn
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses
from models import Bertencoder
from dataset import contraDataset
from torch.utils.tensorboard import SummaryWriter
from scripts.utils import *
import argparse


# params
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')

parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)

parser.add_argument('--max_len',type=int,default=56)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--lr',type=int,default=0.0001)
parser.add_argument('--epoch',type=int,default=40)
parser.add_argument('--exp_name',type=str)

args = parser.parse_args()

model = Bertencoder(args.model_name).to(device=args.device)

trainDS, valDS =  contraDataset(args.train_dir,args.tokenizer_name,args.max_len,args.device), contraDataset(args.val_dir,args.tokenizer_name,args.max_len,args.device)

trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=2)
valDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=2)

optimizer =  optim.Adam( model.parameters() , lr=args.lr, weight_decay=1e-3)
loss_func = losses.TripletMarginLoss()

# training loop
print('*'*10  + 'Training loop started' + '*'*10)


for _ in range(1,args.epoch):

    train_loss = 0.0
    model.train()
    # training loop
    
    start_train = time.time()
    num_batch = 0
    for idx,batch in enumerate(trainDL,0):
        num_batch += 1
        
        text_ids = batch['text_token_ids'].to(args.device, dtype = torch.long)
        text_mask = batch['text_mask'].to(args.device, dtype = torch.long)

        aug_ids = batch['aug_token_ids'].to(args.device, dtype = torch.long)
        aug_mask = batch['aug_mask'].to(args.device, dtype = torch.long)
        
        label = batch['label'].to(args.device, dtype = torch.long)

        embedding_text = model(text_ids,text_mask)
        embedding_aug = model(aug_ids,aug_mask)
        
        optimizer.zero_grad()
        embeddings = torch.cat([embedding_aug,embedding_text],dim=0)
        label = torch.cat([label,label],dim=0)
        contraLoss = loss_func(embedding,label)
        loss.backward()
        optimizer.step()

        epoch_loss += contraLoss.detach()
        #print(loss.detach())
    
    epoch_loss = epoch_loss/float(num_batch)
    end_train = time.time()
    #writer.add_scalar('Loss/train', epoch_loss, _)
    print("Train Epoch: {epoch_no} train_loss: {loss} time elapsed: {time}".format(epoch_no = _ , loss = epoch_loss , time = end_train - start_train))