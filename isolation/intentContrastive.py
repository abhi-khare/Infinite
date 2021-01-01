import time,os 
import torch 
import torch.nn as nn
import pandas as pd
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses, samplers
from dataset import nluDataset
from torch.utils.tensorboard import SummaryWriter
from utils import getSlotsLabels
from arguments import ICPT_arguments
from models import Bertencoder

# CLI arguments
args = ICPT_arguments()

# writer object for tensorboard logs 
writer = SummaryWriter(args.exp_name)

# model class object
model = Bertencoder(args.model_name).to(device=args.device)

# train and validation dataset
trainDS, valDS = nluDataset(args.train_dir,args.tokenizer_name,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_name,args.max_len,args.device)

# train and val labels for MCLass samplers
train_labels = list(pd.read_csv(args.train_dir,sep='\t').intent_ID)
val_labels = list(pd.read_csv(args.val_dir,sep='\t').intent_ID)

# sampler for intent contrastive learning
sampler_train = samplers.MPerClassSampler(train_labels, 6, batch_size=None, length_before_new_iter=30000)
sampler_val = samplers.MPerClassSampler(val_labels, 3, batch_size=None, length_before_new_iter=10000)

# dataset iterators for train and validation set
trainDL = DataLoader(trainDS,batch_size=args.intent_num*6,num_workers=args.num_worker, sampler=sampler_train)
valDL = DataLoader(valDS,batch_size=args.intent_num*6,num_workers=args.num_worker,sampler=sampler_val)

# optimizer and loss function
optimizer =  optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weights)
loss_func = losses.TripletMarginLoss(margin=args.margin)

# validation loop
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


num_iter, best_loss = 0,1000.0

for epoch in range(args.epoch):

    for train_batch in trainDL:
        
        # training loop
        model.train()
        
        text_ids = train_batch['token_ids'].to(args.device, dtype = torch.long)
        text_mask = train_batch['mask'].to(args.device, dtype = torch.long)
        label = train_batch['intent_id'].to(args.device, dtype = torch.long)

        embeddings = model(text_ids,text_mask)
        
        optimizer.zero_grad()
        contraLoss = loss_func(embeddings,label)
        contraLoss.backward()
        optimizer.step()
        
        num_iter += 1

        train_batch_loss = contraLoss.detach()
        writer.add_scalar('train loss', train_batch_loss, num_iter)
        print("Train iteration #: {iter} train_loss: {loss} time elapsed: {time}".format(iter = num_iter , loss = train_batch_loss))

        # validation loop
        if num_iter % 10 == 0:
            val_loss = validation(model,valDL,num_iter)
            writer.add_scalar('val loss', val_loss, num_iter)
            print('iter:',num_iter/10,'val_loss:', val_loss)

            if best_loss > val_loss:
                best_loss = val_loss
                model.encoder.save_pretrained(args.model_export)
