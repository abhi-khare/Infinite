import os, time
import torch
import torch.nn as nn 
from torch import cuda
import pandas as pd
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer
from TorchCRF import CRF
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from seqeval.metrics import f1_score
import optuna
from scripts.utils import *
from dataset import nlu_dataset
from models import jointBert

parser = argparse.ArgumentParser()
###################################################################################################################
# model parameters
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--num_slots', type=int , default=160)
parser.add_argument('--joint_loss_coef', type=float, default=1.0)
parser.add_argument('--freeze_encoder', type=bool , default=False)
parser.add_argument('--shuffle_data', type=bool , default=True)
parser.add_argument('--num_worker', type=int , default=1)

#data
parser.add_argument('--train_dir',type=str)
parser.add_argument('--val_dir',type=str)
parser.add_argument('--num_intent', type=int, default=17)
parser.add_argument('--max_len', type=int, default=56)
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

# training parameters
parser.add_argument('--epoch',type=int,default=40)
parser.add_argument('--batch_size',type=int,default=128)
#parser.add_argument('--weight_decay',type=float,default=0.003)

#misc. 
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_name', type=str)


args = parser.parse_args()

writer = SummaryWriter(args.exp_name)
###################################################################################################################

# loading dataset
trainDS, valDS =  nlu_dataset(args.train_dir,args.tokenizer_weights,args.max_len), nlu_dataset(args.val_dir,args.tokenizer_weights,args.max_len)

def objective(trial):
    model = jointBert(args,trial).to(device=args.device)

    #if args.freeze_encoder:
    #    for params in model.encoder.parameters():
    #        params.requires_grad = False
    
    #instantiate optimizer
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 1e-3, log=True) # lr for encoder
    lr_rest = trial.suggest_float("lr_rest", 1e-5, 1e-3, log=True) # lr for other layer
    optimizer =  optim.Adam([{'params': model.encoder.parameters(), 'lr': lr_encoder}], lr=lr_rest,weight_decay=1e-3)
    

    trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)
    valDL = DataLoader(valDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)

    # training of model
    best_val_loss =1000000

    
    for epoch in range(1,args.epoch):

        model.train()
        for idx,batch in enumerate(trainDL,0):
        
            token_ids = batch['token_ids'].to(args.device, dtype = torch.long)
            mask = batch['mask'].to(args.device, dtype = torch.long)
            intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
            slot_target = batch['slot_target'].to(args.device, dtype = torch.long)
            slot_label = batch['slot_label']
            slot_mask = batch['slot_mask'].to(args.device, dtype = torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()
            joint_loss , slot_pred, intent_pred = model(token_ids,mask,intent_target,slot_target,slot_mask)
            joint_loss.backward()
            optimizer.step()

        
        # validation
        model.eval()
        val_loss, slots_F1 = 0,0
        num_batch = 0
        with torch.no_grad():
            for idx,batch in enumerate(valDL,0):
                num_batch +=1
                token_ids = batch['token_ids'].to(args.device, dtype = torch.long)
                mask = batch['mask'].to(args.device, dtype = torch.long)
                intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
                slot_target = batch['slot_target'].to(args.device, dtype = torch.long)
                slot_label = batch['slot_label']
                slot_mask = batch['slot_mask'].to(args.device, dtype = torch.long)
                
                joint_loss , slot_pred, intent_pred = model(token_ids,mask,intent_target,slot_target,slot_mask)
                slot_target,slot_pred = get_slot_labels(slot_label,slot_pred,slot_dictionary)
                slot_F1 += f1_score(slot_target,slot_pred)

                #intent_acc += accuracy(intent_target,intent_pred)
                val_loss += joint_loss
        
        slots_F1  = slots_F1/float(num_batch)
        #intent_acc = intent_acc/float(len(valDS))
        val_loss = val_loss/float(num_batch)

        print('Best loss:',best_val_loss,'Slots_f1:', slots_F1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        trial.report(best_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

if __name__ == "__main__":
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler,direction="minimize")
    study.optimize(objective, n_trials=100, timeout=300)

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
