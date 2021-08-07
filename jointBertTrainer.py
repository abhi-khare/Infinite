import argparse
import pandas as pd

import torch 
import pytorch_lightning as pl
from   pytorch_lightning import seed_everything, loggers as pl_loggers
from   pytorch_lightning.callbacks import ModelCheckpoint

from scripts.dataset import dataloader
from scripts.model import jointBert
from scripts.utils import accuracy,slot_F1

#jjseed_everything(42, workers=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% command line arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
parser.add_argument("--intent_dropout", type=float)
parser.add_argument("--slots_dropout", type=float)
parser.add_argument('--jointCoef', type=float, default=0.50)

# training params
parser.add_argument('--lr', type=float)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--l2', type=float, default=0.003)
parser.add_argument('--mode', type=str, default='BASELINE')
parser.add_argument('--noise_type', type=str, default='MC')
parser.add_argument('--checkNEpoch', type=int, default=1)

# data params
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--intent_count', type=int)
parser.add_argument('--slots_count',type=int)
parser.add_argument('--dataset',type=str)

#misc params
parser.add_argument('--exp_num', type=str)
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--param_save_dir', type=str)
parser.add_argument('--logging_dir', type=str)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% command line arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_idx2slots(dataset):

    if dataset == 'SNIPS':
        slot_path = '/efs-storage/Infinite/data/SNIPS/slots_list.tsv'
    elif dataset == 'ATIS':
        slot_path = '/efs-storage/Infinite/data/ATIS/slots_list.tsv'

    # loading slot file
    slots_list = pd.read_csv(slot_path,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
    idx2slots  = {idx:slots for idx,slots in enumerate(slots_list)}
    return idx2slots

idx2slots = get_idx2slots(args.dataset)

# ckpt callback config for pytorch lightning
checkpoint_callback = ModelCheckpoint(
    monitor='val_joint_loss', dirpath= args.param_save_dir,
    filename= args.exp_num + '_JB-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1, mode='min',
)

# tensorboard logging
tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)


class jointBertTrainer(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        self.model = jointBert(args)
        self.args = args

    def forward(self, input_ids, attention_mask , intent_target, slots_target):
        return self.model(input_ids, attention_mask , intent_target, slots_target)

    def training_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['supBatch']['token_ids'], batch['supBatch']['mask']
        intent_target,slots_target = batch['supBatch']['intent_id'], batch['supBatch']['slots_id']

        out = self(token_ids,attention_mask,intent_target,slots_target)
        
        self.log('train_joint_loss', out['joint_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_NER_loss', out['ner_loss'], on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']
    
    def validation_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['supBatch']['token_ids'], batch['supBatch']['mask']
        intent_target,slots_target = batch['supBatch']['intent_id'], batch['supBatch']['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        intent_pred, slot_pred = out['intent_pred'], out['slot_pred']
        
        self.log('val_joint_loss', out['joint_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_IC_loss', out['ic_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_NER_loss', out['ner_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_intent_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('slot_f1', slot_F1(slot_pred ,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr = self.args.lr , weight_decay = self.args.l2)


# initialize the dataloader and model
dm = dataloader(args)
model = jointBertTrainer(args)


trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=tb_logger, callbacks=[checkpoint_callback] ,precision=args.precision,max_epochs=args.epoch, check_val_every_n_epoch=args.checkNEpoch)

# model training
trainer.fit(model, dm)