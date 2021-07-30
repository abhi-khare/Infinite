import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import optuna

import time
import argparse
import pandas as pd
from scripts.utils import accuracy, slot_F1
from scripts.dataset import dataloader
from scripts.collatefunc import collate_sup

parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')

# training params
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mode', type=str, default='BASELINE')
# data params
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--intent_count', type=int)
parser.add_argument('--slots_count',type=int)
parser.add_argument('--dataset',type=str)

#misc params
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--logging_dir', type=str)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

def get_idx2slots(dataset):

    if dataset == 'SNIPS':
        slot_path = '/efs-storage/Infinite/data/SNIPS/slots_list.tsv'
    elif dataset == 'ATIS':
        slot_path = '/efs-storage/Infinite/data/ATIS/slots_list.tsv'

    # loading slot file
    slots_list = pd.read_csv(slot_path,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
    idx2slots  = {idx:slots for idx,slots in enumerate(slots_list)}
    return idx2slots

# loading slot index file
idx2slots = get_idx2slots(args.dataset)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class jointBert_model(nn.Module):
    def __init__(self, args, intent_hidden, slots_hidden, intent_dropout, slots_dropout):

        super(jointBert_model, self).__init__()

        self.encoder = AutoModel.from_pretrained(
            args.encoder,
            return_dict=True,
            cache_dir = '/efs-storage/model/',
            output_hidden_states=True,
            sinusoidal_pos_embds=True
        )

        self.intent_head = nn.Sequential(
                                         nn.Dropout(intent_dropout),
                                         nn.Linear(768,args.intent_count)
                                        )

        self.slots_head = nn.Sequential(
                                         nn.Dropout(slots_dropout),
                                         nn.Linear(768,args.slots_count)
                                        )

        self.CE_loss = nn.CrossEntropyLoss()
        self.jointCoef = 0.5
        self.args = args


    def forward(self, input_ids, attention_mask, intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_logits = self.intent_head(intent_hidden)

        # slots data flow
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_head(slots_hidden)

        # accumulating intent classification loss
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        # accumulating slot prediction loss
        slots_loss = self.CE_loss(
            slots_logits.view(-1, self.args.slots_count), slots_target.view(-1)
        )
        slots_pred = torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        joint_loss = self.jointCoef * intent_loss + (1.0 - self.jointCoef) * slots_loss

        return {
            "joint_loss": joint_loss,
            "ic_loss": intent_loss,
            "ner_loss": slots_loss,
            "intent_pred": intent_pred,
            "slot_pred": slots_pred,
        }



def objective(trial: optuna.trial.Trial) -> float:

    timestamp = time.time()

    writer = SummaryWriter(log_dir=args.logging_dir + f'/{str(timestamp)}/')

    # We optimize the number of layers, hidden units in each layer and dropouts.
    idropout = trial.suggest_float("idropout", 0.2, 0.5)
    sdropout = trial.suggest_float("sdropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 0.00001, 0.00006)

    
    model = jointBert_model(args,idropout, sdropout).to(DEVICE)

    dm = dataloader(args)
    dm.setup()
    
    trainDL, valDL = dm.train_dataloader() , dm.val_dataloader()
    
    optimizer = torch.optim.AdamW( model.parameters() , lr = lr , weight_decay = 0.003)
    val_acc , val_slotsF1 = 0.0, 0.0
    # training
    model.train()
    for epoch in range(args.epoch):
        
        for batch in trainDL:
            token_ids, attention_mask = batch["token_ids"].to(DEVICE), batch["mask"].to(DEVICE)
            intent_target, slots_target = batch["intent_id"].to(DEVICE), batch["slots_id"].to(DEVICE)

            out = model(token_ids, attention_mask, intent_target, slots_target)
            loss = out["joint_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #validation
        model.eval()
        acc,slotsF1,cnt = 0.0,0.0,0.0
        with torch.no_grad():
            
            for batch in valDL:

                token_ids, attention_mask = batch["token_ids"].to(DEVICE), batch["mask"].to(DEVICE)
                intent_target, slots_target = batch["intent_id"].to(DEVICE), batch["slots_id"].to(DEVICE)

                out = model(token_ids, attention_mask, intent_target, slots_target)
                intent_pred, slot_pred = out["intent_pred"], out["slot_pred"]
                
                acc += accuracy(intent_pred, intent_target)
                slotsF1 += slot_F1(slot_pred, slots_target, idx2slots)
                cnt += 1.0
        acc = acc/float(cnt)
        slotsF1 = slotsF1/float(cnt)

        writer.add_scalar('acc/val', acc, epoch)
        writer.add_scalar('slotsF1/val', slotsF1, epoch)
        val_acc, val_slotsF1 = acc, slotsF1

    return val_acc, val_slotsF1

sampler = optuna.samplers.MOTPESampler(n_startup_trials=21)
study = optuna.create_study(directions=["maximize","maximize"])
study.optimize(objective, n_trials=40, timeout=100000)

print("Number of finished trials: {}".format(len(study.trials)))
