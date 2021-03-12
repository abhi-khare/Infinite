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
from pytorch_metric_learning import samplers
from dataset import nluDataset
from utils import getSlotsLabels
import argparse


parser = argparse.ArgumentParser()
###################################################################################################################
# model parameters
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

#training parameters 
parser.add_argument('--num_iter',type=int,default=250)
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


args = parser.parse_args()

###################################################################################################################


def accuracy(pred,target):
    return torch.sum(pred==target)/args.batch_size

class Bertencoder(nn.Module):

    def __init__(self,model):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 256)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden


# loading dataset
trainDS, valDS =  nluDataset(args.train_dir,args.tokenizer_weights,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_weights,args.max_len, args.device)

train_labels = list(pd.read_csv('./data/train_EN.tsv',sep='\t').intent_ID)
val_labels = list(pd.read_csv('./data/dev_EN.tsv',sep='\t').intent_ID)

sampler_train = samplers.MPerClassSampler(train_labels, 6, batch_size=None, length_before_new_iter=30000)
sampler_val = samplers.MPerClassSampler(val_labels, 3, batch_size=None, length_before_new_iter=5000)

def objective(trial):

    model = Bertencoder(args.model_weights).to(device=args.device)
    #instantiate optimizer
    
    lr = trial.suggest_float("encoder_lr", 1e-5, 1e-3, log=True) 
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    
    optimizer =  optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    #loss_func = losses.NTXentLoss(temperature=0.07)

    loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    trainDL = DataLoader(trainDS,batch_size=args.intent_num*6,num_workers=args.num_worker, sampler=sampler_train)
    valDL = DataLoader(valDS,batch_size=args.intent_num*6,num_workers=args.num_worker,sampler=sampler_val)

    # training of model
    best_loss = 1000.0
    num_iter = 1
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
        
        # validation

        if num_iter % 20 == 0:
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
            
            print('iter:',num_iter,'val_loss:', val_loss)
            if best_loss > val_loss:
                best_loss = val_loss
            
        trial.report(best_loss, num_iter)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_loss

if __name__ == "__main__":
    
    trial_sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=trial_sampler,direction="minimize")
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