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


basePath = './' 

# loading slot index file
final_slots = pd.read_csv( basePath + "data/SNIPS/slot_list.tsv", sep=",", header=None, names=["SLOTS"]
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

def slot_F1(pred,target,id2slots):

    pred_list = pred.tolist()
    target_list = target.tolist()
    
    pred_slots , target_slots = [],[]

    for idx,sample in enumerate(target_list):
        pred_sample,target_sample = [],[]
        for jdx,slot in enumerate(sample):

            if (slot == -100 or slot==0)!=True:
                target_sample.append(id2slots[slot])
                pred_sample.append(id2slots[pred_list[idx][jdx]])

        pred_slots.append(pred_sample)
        target_slots.append(target_sample)
    
    return f1_score( target_slots, pred_slots,mode='strict', scheme=IOB2)


def accuracy(pred,target):
    return torch.sum(pred==target)/len(target)

class nluDataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):

        self.data = pd.read_csv(file_dir, sep="\t")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.max_len = max_len

    def processSlotLabel(self, word_ids, slot_ids):

        # replace None and repetition with -100
        word_ids = [-100 if word_id == None else word_id for word_id in word_ids]
        previous_word = -100
        for idx, wid in enumerate(word_ids):

            if wid == -100:
                continue
            if wid == previous_word:
                word_ids[idx] = -100
            previous_word = wid

        Pslot_ids = []
        for sid in slot_ids.split():
            Pslot_ids.append(int(sid))
        
        new_labels = [-100 if word_id == -100 else Pslot_ids[word_id] for word_id in word_ids
        return new_labels


    def __getitem__(self, index):

        text = str(self.data.TEXT[index])

        inputs = self.tokenizer.encode_plus(
            text.split(),
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            is_split_into_words=True
        )

        # text encoding
        token_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        word_ids = inputs.word_ids()

        # intent
        intent_id = torch.tensor(self.data.INTENT_ID[index], dtype=torch.long)
        intent_label = self.data.INTENT[index]

        # label processing
        slot_label = self.data.SLOTS[index]
        slot_id = self.processSlotLabel(word_ids, self.data.SLOTS_ID[index])

        slot_id = torch.tensor(slot_id, dtype=torch.long)

        return {
            "token_ids": token_ids,
            "mask": mask,
            "intent_id": intent_id,
            "slots_id": slot_id,
            "intent_label": intent_label,
            "slots_label": slot_label,
            "text": text,
            "slotsID": self.data.SLOTS_ID[index],
        }

    def __len__(self):
        return len(self.data)


class NLU_Dataset_pl(pl.LightningDataModule):
    def __init__(
        self, train_dir, val_dir, test_dir, tokenizer, max_len, batch_size, num_worker
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_worker = num_worker

    def setup(self, stage: [str] = None):
        self.train = nluDataset(self.train_dir, self.tokenizer, self.max_len)

        self.val = nluDataset(self.val_dir, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_worker
        )

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
