import pandas as pd
import optuna
import os,time, pickle 
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from transformers import  DistilBertModel
from TorchCRF import CRF
from torch.utils.data import DataLoader
from torch import cuda
from dataset import nluDataset
from torch.utils.tensorboard import SummaryWriter
from seqeval.metrics import f1_score
from scripts.utils import getSlotsLabels

parser = argparse.ArgumentParser()
###################################################################################################################
# model parameters
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--joint_loss_coef', type=float, default=1.0)
parser.add_argument('--freeze_encoder', type=bool , default=False)

#training parameters 
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--check_val_every_n_epoch',type=int,default=1)
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

#writer = SummaryWriter(args.exp_name)
###################################################################################################################

with open('./notebooks/map_ids_slots.pickle', 'rb') as handle:
    map_idx_slots = pickle.load(handle)

def accuracy(pred,target):
    return torch.sum(pred==target)/args.batch_size

class jointBert(nn.Module):

    def __init__(self, args,trial):

        super(jointBert,self).__init__()
        
        # base encoder
        self.encoder = DistilBertModel.from_pretrained(args.model_weights,return_dict=True,output_hidden_states=True)

        # intent layer
        p_intent = trial.suggest_float("intent_dropout", 0.1, 0.4)
        self.intent_dropout = nn.Dropout(p_intent)
        self.intent_linear_1 = nn.Linear(768, 64)
        self.intent_linear_2 = nn.Linear(64, args.intent_num)
        
        
        # slots layer
        p_slots = trial.suggest_float("slots_dropout", 0.1, 0.4)
        self.slots_dropout = nn.Dropout(p_slots)
        self.slots_classifier_1 = nn.Linear(768, 256)
        self.slots_classifier_2 = nn.Linear(256, args.slots_num)

        self.crf = CRF(args.slots_num)

        self.intent_loss = nn.CrossEntropyLoss(reduction='none')
        #self.joint_loss_coef =  args.joint_loss_coef

        self.log_vars = nn.Parameter(torch.zeros((2)))
    

    
    def forward(self, input_ids, attention_mask, intent_target, slots_target,slots_mask):

        encoded_output = self.encoder(input_ids, attention_mask)

        #intent data flow
        intent_hidden = encoded_output[0][:,0]
        intent_hidden = self.intent_linear_1(self.intent_dropout(F.relu(intent_hidden)))
        intent_logits = self.intent_linear_2(F.relu(intent_hidden))
        # accumulating intent classification loss 
        intent_loss = self.intent_loss(intent_logits, intent_target)

        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)
        
        # slots data flow 
        slots_hidden = encoded_output[0]
        slots_hidden = self.slots_classifier_1(self.slots_dropout(F.relu(slots_hidden)))
        slots_logits = self.slots_classifier_2(F.relu(slots_hidden))

        # accumulating slot prediction loss
        slots_loss = -1 * self.crf(slots_logits, slots_target, mask=slots_mask.byte())
        #slots_loss = torch.mean(slots_loss)

        precision1 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision1 * intent_loss + self.log_vars[0], -1)
        precision2 = torch.exp(-self.log_vars[1])
        loss +=  torch.sum(precision2 * slots_loss + self.log_vars[1], -1)
        
        joint_loss = torch.mean(loss)
        #joint_loss = (slots_loss + intent_loss)/2.0

        slots_pred = self.crf.viterbi_decode(slots_logits, slots_mask.byte())

        return joint_loss,slots_pred,intent_pred



# loading dataset
trainDS, valDS =  nluDataset(args.train_dir,args.tokenizer_weights,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_weights,args.max_len, args.device)

def objective(trial):
    model = jointBert(args,trial).to(device=args.device)
    
    #instantiate optimizer
    encoder_lr = trial.suggest_float("encoder_lr", 1e-5, 1e-3, log=True) # lr for encoder
    intent_lr = trial.suggest_float("intent_lr", 1e-5, 1e-3, log=True) # lr for encoder
    slots_lr = trial.suggest_float("slots_lr", 1e-5, 1e-3, log=True) # lr for other layer
    weight_decay = trial.suggest_float("weight_decay", 0.1, 0.001, log=True)
    optimizer =  optim.Adam([{'params': model.encoder.parameters(), 'lr': encoder_lr},
                             {'params': model.intent_linear_1.parameters(),'lr': intent_lr},
                             {'params': model.intent_linear_2.parameters(),'lr': intent_lr},
                             {'params': model.slots_classifier_1.parameters(),'lr': slots_lr},
                             {'params': model.slots_classifier_2.parameters(),'lr': slots_lr},
                             {'params': model.crf.parameters(),'lr': slots_lr},
                             ], weight_decay=weight_decay)
    

    trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)
    valDL = DataLoader(valDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)

    # training of model
    
    for epoch in range(1,args.epoch):

        model.train()
        for idx,batch in enumerate(trainDL,0):

            token_ids = batch['token_ids'].to(args.device, dtype = torch.long)
            mask = batch['mask'].to(args.device, dtype = torch.long)
            intent_target = batch['intent_id'].to(args.device, dtype = torch.long)
            slots_target = batch['slots_id'].to(args.device, dtype = torch.long)
            slots_label = batch['slots_label']
            slots_mask = batch['slots_mask'].to(args.device, dtype = torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()
            joint_loss , sp, ip = model(token_ids,mask,intent_target,slots_target,slots_mask)
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
                intent_target = batch['intent_id'].to(args.device, dtype = torch.long)
                slots_target = batch['slots_id'].to(args.device, dtype = torch.long)
                slots_label = batch['slots_label']
                slots_mask = batch['slots_mask'].to(args.device, dtype = torch.long)
                
                joint_loss , slots_pred, intent_pred = model(token_ids,mask,intent_target,slots_target,slots_mask)
                slots_target,slots_pred = getSlotsLabels(slots_label,slots_pred,map_idx_slots)
                
                slots_F1 += f1_score(slots_target,slots_pred)
                val_loss += joint_loss.detach()
                intent_acc += accuracy(intent_pred,intent_target)
        
        slots_F1  = slots_F1/float(num_batch)
        intent_acc = intent_acc/float(num_batch)
        val_loss = val_loss/float(num_batch)
        
        print('Best loss:',val_loss,'Slots_f1:', slots_F1 , 'intent_acc:', intent_acc)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

if __name__ == "__main__":
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler,direction="minimize")
    study.optimize(objective, n_trials=100, timeout=3600)

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
