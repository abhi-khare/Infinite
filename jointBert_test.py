import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import  DistilBertModel,DistilBertTokenizerFast

import pandas as pd
import random

from scripts.dataset import *
from scripts.model import IC_NER
from scripts.utils import *
from os import listdir
from os.path import isfile, join
from arguments import test_argument

args = test_argument()


class jointBert(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.IC_NER = IC_NER(args)
        self.args = args

    def forward(self, input_ids, attention_mask , intent_target, slots_target):
        return self.IC_NER(input_ids, attention_mask , intent_target, slots_target)

    def training_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        
        self.log('train_IC_NER_loss', out['joint_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train_NER_loss', out['ner_loss'], on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']
    
    def validation_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        intent_pred, slot_pred = out['intent_pred'], out['slot_pred']
        
        self.log('val_IC_NER_loss', out['joint_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_IC_loss', out['ic_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_NER_loss', out['ner_loss'], on_step=False, on_epoch=True,  logger=True)
        self.log('val_intent_acc', accuracy(out['intent_pred'],intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('slot_f1', slot_F1(out['slot_pred'],slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
        
        
        return out['joint_loss']
    
    def test_step(self,batch,batch_idx):
        
        token_ids, attention_mask = batch['token_ids'], batch['mask']
        intent_target,slots_target = batch['intent_id'], batch['slots_id']
        
        out = self(token_ids,attention_mask,intent_target,slots_target)
        intent_pred, slot_pred = out['intent_pred'], out['slot_pred']
        
        self.log('test_intent_acc', accuracy(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)
        self.log('test_slot_f1', slot_F1(slot_pred,slots_target,idx2slots), on_step=False, on_epoch=True, logger=True)
        
        return out['joint_loss']
        

    def configure_optimizers(self):
         return torch.optim.AdamW( self.parameters(), lr=3e-5 ,  weight_decay=args.weight_decay)



def cal_mean_stderror(metric):
    var,std_error = 0,0
    mean = sum(metric)/len(metric)
    for m in metric:
        var += (m-mean)**2
    var = (var/(len(metric)-1))**0.5
    std_error = var/((len(metric))**0.5)
    return [round(mean,4),round(std_error,4)]

# fetching slot-idx dictionary
final_slots = pd.read_csv('./data/multiATIS/slots_list.csv',sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
idx2slots  = {idx:slots for idx,slots in enumerate(final_slots)}

test_dir = args.test_dir 
testSet_Path = [join(test_dir,f) for f in listdir(test_dir) if isfile(join(test_dir, f))]

model_dir = args.model_dir 
model_Path = [join(model_dir,f) for f in listdir(model_dir) if isfile(join(model_dir, f))]

print(testSet_Path,model_Path)

trainer = pl.Trainer(gpus=-1,precision=16,accumulate_grad_batches=4,max_epochs=15, check_val_every_n_epoch=1)


for test_file in testSet_Path:
    
    dl = NLU_Dataset_pl(test_file,test_file,test_file , 'distilbert-base-multilingual-cased',56,1)
    dl.setup()
    testLoader = dl.test_dataloader()
    

    acc,slotF1 = [],[]

    for model_path in model_Path:

        model = jointBert.load_from_checkpoint(checkpoint_path=model_Path,map_location=None)
        model.eval()
        print('yesssss')

        out = trainer.test(model=model,test_dataloaders=testLoader)

        acc.append(out[0]['test_intent_acc'])
        slotF1.append(out[0]['test_slot_f1'])
    

    print('test_file: ', test_file ,'acc:',cal_mean_stderror(acc),'slotsF1',cal_mean_stderror(slotF1))
