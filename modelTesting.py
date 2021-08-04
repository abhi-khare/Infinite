import os 
import pandas as pd
import argparse
import torch 

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from functools import partial
from transformers import AutoTokenizer
from scripts.model import jointBert
from scripts.dataset import dataset
from scripts.utils import accuracy, slot_F1, cal_mean_stderror
from scripts.collatefunc import collate_sup
from data.testTemplate import test_template



parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
parser.add_argument("--intent_dropout", type=float)
parser.add_argument("--slots_dropout", type=float)
parser.add_argument('--jointCoef', type=float, default=0.50)

# training params
parser.add_argument('--lr', type=float)
parser.add_argument('--l2', type=float, default=0.003)

# data params
parser.add_argument('--test_path',type=str)
parser.add_argument('--intent_count', type=int)
parser.add_argument('--slots_count',type=int)
parser.add_argument('--dataset',type=str)

#misc params
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--base_dir',type=str)

args = parser.parse_args()

seed_everything(42, workers=True)

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

    def test_step(self,batch,batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        intent_pred, slot_pred = out['intent_pred'], out['slot_pred']

        self.log('test_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('test_slotsF1', slot_F1(slot_pred,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr = self.args.lr , weight_decay = self.args.l2)

trainer = pl.Trainer(gpus=args.gpus, deterministic=True,precision=args.precision)

testSet = test_template(args.dataset, args.test_path)

# testing the model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,cache_dir = '/efs-storage/tokenizer/')          
model_path = [ args.base_dir + name for name in os.listdir(args.base_dir)]

for path in model_path:

    model = jointBertTrainer.load_from_checkpoint(path,args=args)

    for testName,test in testSet.items():

        acc,slotF1 = [],[]
        print('calculating metrics for the testset: ', testName)

        for test_file in test[0]:
        
            testDS = dataset(test_file)
            testDL = DataLoader( testDS, batch_size=test[1], collate_fn=partial(collate_sup,tokenizer = tokenizer))

            out = trainer.test(model=model ,test_dataloaders=testDL)
            
            acc.append(out[0]['test_acc'])
            slotF1.append(out[0]['test_slotsF1'])
        
        print('acc: ',cal_mean_stderror(acc),'slotsF1: ',cal_mean_stderror(slotF1))