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
parser.add_argument('--max_len', type=int, default=46)
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')

# training parameters
parser.add_argument('--epoch',type=int,default=60)
parser.add_argument('--batch_size',type=int,default=64)
#parser.add_argument('--weight_decay',type=float,default=0.003)

#misc. 
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_name', type=str)


args = parser.parse_args()

writer = SummaryWriter(args.exp_name)
###################################################################################################################

def process_label(labels, max_len):
    slot_target = []
        
    for sLabel in labels:
        slots = [int(L) for L in sLabel.split()]
        slots += [160]*(max_len - len(slots))
        slot_target.append(slots)
        
    slot_target = torch.LongTensor(slot_target)
    return slot_target

def accuracy(pred,target):
    return torch.sum(pred==target)/float(len(pred))

class jointBert(nn.Module):

    def __init__(self, args,trial):

        super(jointBert,self).__init__()
        
        # base encoder
        self.encoder = DistilBertModel.from_pretrained(args.model_weights,return_dict=True,output_hidden_states=True)

        # intent layer
        p_intent = trial.suggest_float("dropout_intent", 0.1, 0.4)
        self.intent_dropout = nn.Dropout(p_intent)
        self.intent_linear = nn.Linear(768, args.num_intent)
        
        # slots layer
        self.slot_classifier = nn.Linear(768, args.num_slots)
        p_slots = trial.suggest_float("dropout_slots", 0.1, 0.4)
        self.dropout_slots = nn.Dropout(p_slots)

        self.crf = CRF(args.num_slots)

        self.intent_loss = nn.CrossEntropyLoss()
        self.joint_loss_coef =  args.joint_loss_coef
    

    
    def forward(self, input_ids, attention_mask, intent_target, slot_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        #intent data flow
        intent_hidden = encoded_output[0][:,0]
        intent_hidden = self.intent_dropout(intent_hidden)
        intent_logits = self.intent_linear(intent_hidden)
        # accumulating intent classification loss 
        intent_loss = self.intent_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits))
        
        # slots data flow 
        slots_hidden = encoded_output[0]
        slots_logits = self.slot_classifier(self.dropout_slots(slots_hidden))
        # accumulating slot prediction loss
        slot_loss = -1 * self.joint_loss_coef * self.crf(slots_logits, slot_target, mask=attention_mask.byte())
        slot_loss = torch.mean(slot_loss)
        
        joint_loss = (slot_loss + intent_loss)/2.0

        slot_pred = self.crf.viterbi_decode(slots_logits, attention_mask.byte())

        return joint_loss,slot_pred,intent_pred


class nlu_dataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
    def __getitem__(self, index):
        
        text = str(self.data.utterance[index])
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
        
        token_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'id' : torch.tensor(self.data.id[index],dtype=torch.long),
            'intent_label': self.data.intent[index],
            'slot_label' : self.data.slot_labels[index],
            'intent_target': torch.tensor(self.data.intent_ID[index], dtype=torch.long),
            'slot_target' : self.data.slots_ID[index],
            'language' : self.data.language[index]
        } 
    
    def __len__(self):
        return len(self.data)

# loading dataset
trainDS, valDS =  nlu_dataset(args.train_dir,args.tokenizer_weights,args.max_len), nlu_dataset(args.val_dir,args.tokenizer_weights,args.max_len)

def objective(trial):
    model = jointBert(args,trial).to(device=args.device)

    if args.freeze_encoder:
        for params in model.encoder.parameters():
            params.requires_grad = False
    
    #instantiate optimizer
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 1e-3, log=True) # lr for encoder
    lr_rest = trial.suggest_float("lr_rest", 1e-5, 1e-3, log=True) # lr for other layer
    optimizer =  optim.Adam([{'params': model.encoder.parameters(), 'lr': lr_encoder}], lr=lr_rest,weight_decay=1e-3)
    

    trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)
    valDL = DataLoader(valDS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)

    # training of model
    #val_intent_acc =0
    
    for _ in range(1,args.epoch):

        model.train()
        for idx,batch in enumerate(trainDL,0):
        
            ids = batch['ids'].to(args.device, dtype = torch.long)
            mask = batch['mask'].to(args.device, dtype = torch.long)
            intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
            slot_target = process_label(batch['slot_target'],args.max_len).to(args.device, dtype = torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss,slot_pred,intent_pred = model(ids,mask,intent_target,slot_target)
            loss.backward()
            optimizer.step()

        
        # validation
        model.eval()
        intent_acc , slots_F1 = 0,0
        with torch.no_grad():
            for idx,batch in enumerate(valDL,0):
                
                ids = batch['ids'].to(args.device, dtype = torch.long)
                mask = batch['mask'].to(args.device, dtype = torch.long)
                intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
                slot_target = process_label(batch['slot_target'],args.max_len).to(args.device, dtype = torch.long)

                loss,slot_pred,intent_pred = model(ids,mask,intent_target,slot_target)
                slots_F1 += f1_score(slot_target.numpy().tolist(),slot_pred)
                intent_acc += accuracy(intent_target,intent_pred)
        
        slots_F1  = slots_F1/float(len(valDS))
        intent_acc = intent_acc/float(len(valDS))
        trial.report(accuracy, _)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, timeout=300)

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
