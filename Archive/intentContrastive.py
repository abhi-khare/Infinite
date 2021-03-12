import time,os 
import torch 
import torch.nn as nn
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses
from models import Bertencoder
from dataset import nluDataset
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
parser.add_argument('--model_export', type=str)

parser.add_argument('--max_len',type=int,default=56)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--lr',type=int,default=0.0001)
parser.add_argument('--weights',type=int,default=0.0001)
parser.add_argument('--epoch',type=int,default=40)
parser.add_argument('--exp_name',type=str)

args = parser.parse_args()

writer = SummaryWriter('logs/PT/'+args.exp_name)

model = Bertencoder(args.model_name).to(device=args.device)

trainDS, valDS = nluDataset(args.train_dir,args.tokenizer_name,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_name,args.max_len,args.device)

trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=2)
valDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=2)

optimizer =  optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weights)

loss_func = losses.TripletMarginLoss(margin=2.0)

def validation(model,val_DL,epoch):

    # validation loop
    start_val = time.time()
    model.eval()
    val_loss ,num_batch = 0.0,0
    with torch.no_grad():
        for _,batch in enumerate(val_DL,0):
            
            num_batch +=1
            text_ids = batch['token_ids'].to(args.device, dtype = torch.long)
            text_mask = batch['mask'].to(args.device, dtype = torch.long)
            labels = batch['intent_id'].to(args.device, dtype = torch.long)

            embeddings = model(text_ids,text_mask)
            contraLoss = loss_func(embeddings,labels)
            val_loss += contraLoss.detach()

    end_val = time.time()
    val_loss = val_loss/float(num_batch)

    writer.add_scalar('Loss/val', val_loss, epoch )
    print("Val : val_loss: {loss} time elapsed: {time}".format(loss = val_loss , time = end_val - start_val))


# training loop

for epoch in range(1,args.epoch):
    print('*'*10  + 'Training loop started' + '*'*10)
    train_loss = 0.0
    model.train()
    # training loop
    start_train = time.time()
    num_batch = 0
    for idx,batch in enumerate(trainDL,0):
        num_batch += 1
        
        text_ids = batch['token_ids'].to(args.device, dtype = torch.long)
        text_mask = batch['mask'].to(args.device, dtype = torch.long)
        label = batch['label'].to(args.device, dtype = torch.long)

        embeddings = model(text_ids,text_mask)
        
        optimizer.zero_grad()
        contraLoss = loss_func(embeddings,label)
        contraLoss.backward()
        optimizer.step()

        train_loss += contraLoss.detach()
        
    
    train_loss = train_loss/float(num_batch)
    end_train = time.time()
    writer.add_scalar('Loss/train', train_loss, epoch)
    print("Train Epoch: {epoch_no} train_loss: {loss} time elapsed: {time}".format(epoch_no = epoch , loss = train_loss , time = end_train - start_train))
    
    print('*'*10  + 'Validation loop started' + '*'*10)
    validation(model,valDL)

model.encoder.save_pretrained(args.model_export)