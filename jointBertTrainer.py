import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import  DistilBertTokenizerFast

import pandas as pd

from scripts.dataset import *
from scripts.model import jointBert
from arguments import jointBert_params
seed_everything(42, workers=True)

import argparse

def jointBert_params():

    parser = argparse.ArgumentParser()
    
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
    parser.add_argument("--intHidden", type=int, nargs="+")
    parser.add_argument("--slotHidden", type=int, nargs="+")
    parser.add_argument("--intDrop", type=float, nargs="+")
    parser.add_argument("--slotDrop", type=float, nargs="+")
    
    # training params
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--l2', type=float, default=0.003)
    
    # data params
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--valDir', type=str)
    parser.add_argument('--intCount', type=int)
    parser.add_argument('--slotCount',type=int)

    #misc params
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--paramDir', type=str)
    parser.add_argument('--logDir', type=str)
    parser.add_argument('--precision', type=int, default=16)

    return parser.parse_args()

args = jointBert_params()

if args.data == 'SNIPS':
    slotPath = './data/SNIPS/slot_list.tsv'
elif args.data == 'ATIS':
    slotPath = './data/ATIS/slot_list.tsv'

# loading slot file
slotsList = pd.read_csv(slotPath,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
idx2slots  = {idx:slots for idx,slots in enumerate(slotsList)}

# ckpt callback config for pytorch lightning
checkpoint_callback = ModelCheckpoint(
    monitor='val_joint_loss', dirpath= args.ckpt_dir,
    filename='jointBert-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1, mode='min',
)

# tensorboard logging
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)


class jointBertTrainer(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        self.model = jointBert(args)
        self.args = args

    def forward(self, input_ids, attention_mask , intent_target, slots_target):
        return self.model(input_ids, attention_mask , intent_target, slots_target)

    def training_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        
        self.log('train_joint_loss', out['joint_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_NER_loss', out['ner_loss'], on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']
    
    def validation_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        intent_pred, slot_pred = out['intent_pred'], out['slot_pred']
        
        self.log('val_joint_loss', out['joint_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_IC_loss', out['ic_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_NER_loss', out['ner_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_intent_acc', accuracy(out['intent_pred'],intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('slot_f1', slot_F1(out['slot_pred'],slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']
        

    def configure_optimizers(self):
         return torch.optim.AdamW([
                {'params': self.IC_NER.encoder.parameters() , 'lr' : 0.00002 , 'weight_decay': args.l2},
                {'params': self.IC_NER.intent_FC1.parameters(), 'lr': 0.0003780459727740985},
                {'params': self.IC_NER.intent_FC2.parameters(), 'lr': 0.0003780459727740985},
                {'params': self.IC_NER.slots_FC.parameters(), 'lr': 0.0001748742430446569}])


# initialize the dataloader
dm = NLU_Dataset_pl(args.train_dir, args.val_dir, args.tokenizer, args.max_len, args.batch_size,args.num_worker)

# initialize the model and trainer
model = jointBertTrainer(args)
trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=tb_logger, callbacks=[checkpoint_callback] ,precision=args.precision,max_epochs=args.epoch, check_val_every_n_epoch=args.checkNEpoch)

# model training
trainer.fit(model, dm)