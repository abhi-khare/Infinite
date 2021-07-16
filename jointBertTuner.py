import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import optuna

import argparse
import pandas as pd
from scripts.utils import accuracy, slot_F1, get_idx2slots
from scripts.dataset import dataloader


parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')

# training params
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--l2', type=float, default=0.003)

# data params
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--intent_count', type=int)
parser.add_argument('--slots_count',type=int)
parser.add_argument('--dataset',type=str)

#misc params
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--param_save_dir', type=str)
parser.add_argument('--logging_dir', type=str)
parser.add_argument('--precision', type=int, default=16)

args = parser.parse_args()

# loading slot index file
idx2slots = get_idx2slots(args.dataset)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class jointBert_model(nn.Module):
    def __init__(self, args, intent_head_size, idropout, sdropout):

        super(jointBert_model, self).__init__()

        self.encoder = AutoModel.from_pretrained(
            args.encoder,
            return_dict=True,
            output_hidden_states=True,
            sinusoidal_pos_embds=True
        )

        self.intent_dropout = nn.Dropout(idropout)
        self.intent_hidden = nn.Linear(768, intent_head_size),
        self.intent_head = nn.Linear(intent_head_size, args.intent_count),
                                               
        self.slots_dropout = nn.Dropout(sdropout)
        self.slots_head = nn.Linear(768, args.slots_count)

        self.CE_loss = nn.CrossEntropyLoss()
        self.joint_loss_coef = 0.5


    def forward(self, input_ids, attention_mask, intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_hidden = self.intent_hidden(self.intent_dropout(F.gelu(intent_hidden)))
        intent_logits = self.intent_head(self.intent_dropout(F.gelu(intent_hidden)))

        # slots data flow
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_head(self.slots_dropout(F.relu(slots_hidden)))

        # accumulating intent classification loss
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        # accumulating slot prediction loss
        slots_loss = self.CE_loss(slots_logits.view(-1, self.args.slots_count), slots_target.view(-1))
        slots_pred = torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        joint_loss = self.joint_loss_coef * intent_loss + (1.0 - self.joint_loss_coef) * slots_loss

        return {
            "joint_loss": joint_loss,
            "ic_loss": intent_loss,
            "ner_loss": slots_loss,
            "intent_pred": intent_pred,
            "slot_pred": slots_pred,
        }


trial_cnt = 0

def objective(trial: optuna.trial.Trial) -> float:

    writer = SummaryWriter(log_dir=args.logging_dir + f'/{str(trial_cnt)}/')

    # We optimize the number of layers, hidden units in each layer and dropouts.
    ihidden_size = trial.suggest_int("intent_hidden_size", 64, 512)
    idropout = trial.suggest_float("idropout", 0.2, 0.5)
    sdropout = trial.suggest_float("sdropout", 0.2, 0.5)
    lr = trial.suggest_float("LR", 0.00006, 0.0006)
    l2 = trial.suggest_float("LR", 0.0003, 0.03)

    
    model = jointBert_model(args, ihidden_size, idropout, sdropout).to(DEVICE)

    dm = dataloader(args)
    dm.setup()
    
    trainDL, valDL = dm.train_dataloader() , dm.val_dataloader()
    
    optimizer = torch.optim.AdamW( model.parameters() , lr = lr , weight_decay = l2)

    best_acc,best_F1 = 0.0,0.0
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
        
        if epoch%3==0:

            #validation
            model.eval()
            acc,slotsF1,cnt = 0.0,0.0,0
            with torch.no_grad():
                
                for batch in valDL:

                    token_ids, attention_mask = batch["token_ids"].to(DEVICE), batch["mask"].to(DEVICE)
                    intent_target, slots_target = batch["intent_id"].to(DEVICE), batch["slots_id"].to(DEVICE)

                    out = model(token_ids, attention_mask, intent_target, slots_target)
                    intent_pred, slot_pred = out["intent_pred"], out["slot_pred"]
                    
                    acc += accuracy(intent_pred, intent_target)
                    slotsF1 += slot_F1(slot_pred, slots_target, idx2slots)
                    cnt += 1
            
            acc = acc/float(cnt)
            slotsF1 = slotsF1/float(cnt)

            if acc>best_acc:
                best_acc = acc
                best_F1 = slotsF1

            writer.add_scalar('acc/val', acc, epoch/3)
            writer.add_scalar('slotsF1/val', slotsF1, epoch/3)

    return best_acc, best_F1


sampler = optuna.samplers.MOTPESampler(n_startup_trials=21)
study = optuna.create_study(directions=["maximize","maximize"])
study.optimize(objective, n_trials=40, timeout=100000)

print("Number of finished trials: {}".format(len(study.trials)))
