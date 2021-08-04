import argparse
import pandas as pd

import torch 
import pytorch_lightning as pl
from   pytorch_lightning import seed_everything, loggers as pl_loggers
from   pytorch_lightning.callbacks import ModelCheckpoint
from transformers.utils.dummy_pt_objects import AutoModel

from scripts.dataset import dataloader
from scripts.model import hierCon_model
from scripts.utils import accuracy,slot_F1

seed_everything(42, workers=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% command line arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
parser.add_argument("--intent_dropout", type=float)
parser.add_argument("--slots_dropout", type=float)
parser.add_argument('--jointCoef', type=float, default=0.50)
parser.add_argument('--icnerCoef', type=float, default=0.50)
parser.add_argument('--hierConCoef', type=float, default=0.50)
parser.add_argument("--intent_contrast_hidden", type=int)
parser.add_argument("--slots_contrast_hidden", type=int)


# training params
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--l2', type=float, default=0.003)
parser.add_argument('--mode', type=str, default='BASELINE')
parser.add_argument('--noise_type', type=str, default='MC')
parser.add_argument('--checkNEpoch', type=int, default=1)
parser.add_argument('--warmup',type=int,default=1000)

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
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

# loading slot index file
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
    filename='HierCon-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1, mode='min',
)

tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)


class hierConTrainer(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        self.model = hierCon_model(args)
        self.args = args
        self.step = 0

    def forward(self, batch,mode):
        return self.model(batch,mode)
    
    def training_step(self, batch, batch_idx):

        self.step += 1
        ICNERLoss,hierConLoss = 0.0,0.0
        
        # generating HierCon loss
        try:
            jointCLLoss = self(batch,'hierCon')
            self.log('jointCLLoss', jointCLLoss, on_step=False, on_epoch=True, logger=True)
        
            if self.step <= args.warmup:
                return jointCLLoss
        except:
            a = 1
            
        # generating joint ICNER and HierCon loss
        ICNER_out = self(batch,'ICNER')
        IC_loss, NER_loss = ICNER_out['ic_loss'],ICNER_out['ner_loss']
        
        self.log('IC_loss', IC_loss, on_step=False, on_epoch=True, logger=True)
        self.log('NER_loss', NER_loss, on_step=False, on_epoch=True, logger=True)
        
        return self.args.jointcoef*ICNER_out['joint_loss'] + (1.0 - self.args.jointcoef)*jointCLLoss

    
    def validation_step(self, batch, batch_idx):
        
        ICNER_out = self(batch,'ICNER')
        intent_pred, slot_pred = ICNER_out['intent_pred'], ICNER_out['slot_pred']
        intent_target , slots_target = batch['supBatch']['intent_id'] , batch['supBatch']['slots_id']

        self.log('val_joint_loss', ICNER_out['joint_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_IC_loss', ICNER_out['ic_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_NER_loss', ICNER_out['ner_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_intent_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('slot_f1', slot_F1(slot_pred ,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
        
        return ICNER_out['joint_loss']
    
    def test_step(self, batch, batch_idx):
        
        ICNER_out = self(batch,'ICNER')
        intent_pred, slot_pred = ICNER_out['intent_pred'], ICNER_out['slot_pred']
        intent_target , slots_target = batch['supBatch']['intent_id'] , batch['supBatch']['slots_id']

        self.log('test_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('test_slotF1', slot_F1(slot_pred ,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
    
    def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr = args.lr , weight_decay = args.l2)



# initialize the dataloader and model
dm = dataloader(args)
model = hierConTrainer(args)

trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=tb_logger, callbacks=[checkpoint_callback] ,precision=args.precision,max_epochs=args.epoch, check_val_every_n_epoch=args.checkNEpoch)

# model training
trainer.fit(model, dm)