import argparse
import pandas as pd

import torch 
import pytorch_lightning as pl
from   pytorch_lightning import seed_everything, loggers as pl_loggers
from   pytorch_lightning.callbacks import ModelCheckpoint

from scripts.dataset import dataloader
from scripts.model import jointBert
from scripts.utils import accuracy,slot_F1,get_idx2slots

seed_everything(42, workers=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% command line arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')

# training params
parser.add_argument('--epoch', type=int, default=20)
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
parser.add_argument('--logging_dir', type=str)

args = parser.parse_args()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% command line arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx2slots = get_idx2slots(args.dataset)

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
        self.log('val_intent_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('slot_f1', slot_F1(slot_pred ,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']

    def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr = args.lr , weight_decay = args.l2)


# initialize the dataloader and model
dm = dataloader(args)
model = jointBertTrainer(args)


trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=tb_logger, callbacks=[checkpoint_callback] ,precision=args.precision,max_epochs=args.epoch, check_val_every_n_epoch=args.checkNEpoch)

# model training
trainer.fit(model, dm)