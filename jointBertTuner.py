import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertModel, DistilBertTokenizerFast
import pytorch_lightning as pl
import pandas as pd
import random

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.scheme import IOB2

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# loading slot index file
final_slots = pd.read_csv( "./data/SNIPS/slot_list.tsv", sep=",", header=None, names=["SLOTS"]
).SLOTS.values.tolist()
idx2slots = {idx: slots for idx, slots in enumerate(final_slots)}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model parameter
config = {
    "mc": {
        "model_name": "distilbert-base-cased",
        "tokenizer_name": "distilbert-base-cased",
        "joint_loss_coef": 0.5,
    },
    # training parameters
    "tc": {
        "encoder_lr": 0.00002,
        "epoch": 15,
        "batch_size": 64,
        "weight_decay": 0.003,
        "shuffle_data": True,
        "num_worker": 2
    },
    # data params
    "dc": {
        "train_dir": basePath +  "data/SNIPS/experiments/train/clean/train.tsv",
        "val_dir": basePath + "data/SNIPS/experiments/dev/clean/dev.tsv",
        "max_len": 56,
    },
    # misc
    "misc": {
        "fix_seed": False,
        "gpus": -1,
        "precision": 16,
    },
}


class IC_NER(nn.Module):
    def __init__(self, idropout_1, idropout_2, sdropout, ihidden_size):

        super(IC_NER, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
            config['mc']['model_name'],
            return_dict=True,
            output_hidden_states=True,
            sinusoidal_pos_embds=True
        )

        self.intent_dropout_1 = nn.Dropout(idropout_1)
        self.intent_dropout_2 = nn.Dropout(idropout_2)
        self.intent_FC1 = nn.Linear(768, ihidden_size)
        self.intent_FC2 = nn.Linear(ihidden_size, 8)

        # slots layer
        self.slots_dropout = nn.Dropout(sdropout)
        self.slots_FC = nn.Linear(768, 72)

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss()

        self.jlc = 0.5
        # self.cfg = cfg

    def forward(self, input_ids, attention_mask, intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_hidden = self.intent_FC1(self.intent_dropout_1(F.gelu(intent_hidden)))
        intent_logits = self.intent_FC2(self.intent_dropout_2(F.gelu(intent_hidden)))

        # accumulating intent classification loss
        intent_loss = self.intent_loss_fn(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        # slots data flow
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_FC(self.slots_dropout(F.relu(slots_hidden)))
        slot_pred = torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        # accumulating slot prediction loss
        slot_loss = self.slot_loss_fn(slots_logits.view(-1, 72), slots_target.view(-1))

        joint_loss = self.jlc * intent_loss + (1.0 - self.jlc) * slot_loss

        return {
            "joint_loss": joint_loss,
            "ic_loss": intent_loss,
            "ner_loss": slot_loss,
            "intent_pred": intent_pred,
            "slot_pred": slot_pred,
        }


trial_cnt = 0

def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    ihidden_size = trial.suggest_int("intent_hidden_size", 64, 512)
    
    idropout_1 = trial.suggest_float("idropout1", 0.2, 0.5)
    idropout_2 = trial.suggest_float("idropout2", 0.2, 0.5)
    sdropout = trial.suggest_float("sdropout", 0.2, 0.5)

    intentLR = trial.suggest_float("intentLR", 0.00001, 0.001)
    slotsLR = trial.suggest_float("slotsLR", 0.00001, 0.001)


    
    model = IC_NER(idropout_1, idropout_2, sdropout, ihidden_size).to(DEVICE)

    dm = NLU_Dataset_pl(
        config["dc"]["train_dir"],
        config["dc"]["val_dir"],
        config["mc"]["tokenizer_name"],
        config["dc"]["max_len"],
        config["tc"]["batch_size"],
        config["tc"]["num_worker"],
    )
    dm.setup()
    
    trainDL, valDL = dm.train_dataloader() , dm.val_dataloader()
    
    optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters() , 'lr' : 0.00005 , 'weight_decay': config["tc"]["weight_decay"]},
                {'params': model.intent_FC1.parameters(), 'lr': intentLR},
                {'params': model.intent_FC2.parameters(), 'lr': intentLR},
                {'params': model.slots_FC.parameters(), 'lr': slotsLR}])

    # training
    # training
    model.train()
    for epoch in range(config['tc']['epoch']):
        
        for batch in trainDL:
            token_ids, attention_mask = batch["token_ids"].to(DEVICE), batch["mask"].to(DEVICE)
            intent_target, slots_target = batch["intent_id"].to(DEVICE), batch["slots_id"].to(DEVICE)

            out = model(token_ids, attention_mask, intent_target, slots_target)
            optimizer.zero_grad()
            out["joint_loss"].backward()
            optimizer.step()
            
    
    model.eval()
    
    #validation

    acc,slotsF1,cnt = 0.0,0.0,0
    with torch.no_grad():
        
        for batch in valDL:

            token_ids, attention_mask = batch["token_ids"].to(DEVICE), batch["mask"].to(DEVICE)
            intent_target, slots_target = batch["intent_id"].to(DEVICE), batch["slots_id"].to(DEVICE)

            out = model(token_ids, attention_mask, intent_target, slots_target)
            intent_pred, slot_pred = out["intent_pred"], out["slot_pred"]
            
            acc += accuracy(out["intent_pred"], intent_target)
            slotsF1 += slot_F1(out["slot_pred"], slots_target, idx2slots)
            cnt += 1
        
    acc = acc/float(cnt)
    slotsF1 = slotsF1/float(cnt)

    return acc, slotsF1


sampler = optuna.samplers.MOTPESampler(n_startup_trials=21)
study = optuna.create_study(directions=["maximize","maximize"])
study.optimize(objective, n_trials=30, timeout=100000)

print("Number of finished trials: {}".format(len(study.trials)))
