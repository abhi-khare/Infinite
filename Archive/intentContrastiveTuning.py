import pandas as pd
import optuna
import time,os 
import torch 
import torch.nn as nn
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses
from dataset import nluDataset
from scripts.utils import getSlotsLabels
import argparse


parser = argparse.ArgumentParser()
###################################################################################################################
# model parameters
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

#training parameters 
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--shuffle_data', type=bool , default=True)
parser.add_argument('--num_worker', type=int , default=4)

# data
parser.add_argument('--train_dir',type=str)
parser.add_argument('--val_dir',type=str)
parser.add_argument('--intent_num', type=int, default=17)
parser.add_argument('--slots_num', type=int , default=160)
parser.add_argument('--max_len', type=int, default=56)

#misc. 
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_name', type=str)


args = parser.parse_args()

###################################################################################################################


def accuracy(pred,target):
    return torch.sum(pred==target)/args.batch_size

class Bertencoder(nn.Module):

    def __init__(self,model):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 768)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden


# loading dataset
trainDS, valDS =  nluDataset(args.train_dir,args.tokenizer_weights,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_weights,args.max_len, args.device)

def objective(trial):

    model = Bertencoder(args.model_weights).to(device=args.device)
    #instantiate optimizer
    
    lr = trial.suggest_float("rest_lr", 1e-5, 1e-3, log=True) 
    weight_decay = trial.suggest_float("weight_decay", 1e-1, 1e-3, log=True)
    
    optimizer =  optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    loss_func = losses.TripletMarginLoss(margin=2.0)

    trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)
    valDL = DataLoader(valDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)

    # training of model
    
    for epoch in range(1,args.epoch):

        model.train()
        for idx,batch in enumerate(trainDL,0):
            num_batch += 1
            
            text_ids = batch['token_ids'].to(args.device, dtype = torch.long)
            text_mask = batch['mask'].to(args.device, dtype = torch.long)
            label = batch['intent_id'].to(args.device, dtype = torch.long)

            embeddings = model(text_ids,text_mask)
            
            optimizer.zero_grad()
            contraLoss = loss_func(embeddings,label)
            contraLoss.backward()
            optimizer.step()
        
        # validation
        model.eval()
        val_loss ,num_batch = 0.0,0
        with torch.no_grad():
            for _,batch in enumerate(valDL,0):
                
                num_batch +=1
                text_ids = batch['token_ids'].to(args.device, dtype = torch.long)
                text_mask = batch['mask'].to(args.device, dtype = torch.long)
                labels = batch['intent_id'].to(args.device, dtype = torch.long)

                embeddings = model(text_ids,text_mask)
                contraLoss = loss_func(embeddings,labels)
                val_loss += contraLoss.detach()

        val_loss = val_loss/float(num_batch)  
        
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

if __name__ == "__main__":
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler,direction="minimize")
    study.optimize(objective, n_trials=100, timeout=18000)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
