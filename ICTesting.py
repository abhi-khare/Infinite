import os 
import pandas as pd
import argparse
import torch 

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from functools import partial
from transformers import DistilBertTokenizerFast
from scripts.model import intent_classifier
from scripts.dataset_scripts import dataset
from scripts.utils import F1, cal_mean_stderror
from scripts.collatefunc import collate_sup

from data.testTemplate import test_template
from arguments import ICTesting_args

args = ICTesting_args()

seed_everything(42, workers=True)

class ICTrainer(pl.LightningModule):
    
    def __init__(self, args):
        
        super().__init__()
        
        self.model = intent_classifier(args)
        self.args = args

    def forward(self, input_ids, attention_mask , intent_target):

        return self.model(input_ids, attention_mask , intent_target)

    def training_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['supBatch']['token_ids'], batch['supBatch']['mask']
        intent_target = batch['supBatch']['intent_id']

        out = self(token_ids,attention_mask,intent_target)
        
        self.log('train_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        
        return out['ic_loss']
    
    def test_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['supBatch']['token_ids'], batch['supBatch']['mask']
        intent_target = batch['supBatch']['intent_id']
        
        out = self(token_ids,attention_mask,intent_target)
        intent_pred = out['intent_pred']
        
        self.log('test_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('test_intent_F1', F1(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)

    def configure_optimizers(self):

         return torch.optim.AdamW(self.parameters(), lr = self.args.lr , weight_decay = self.args.l2)


trainer = pl.Trainer(gpus=args.gpus, deterministic=True,precision=args.precision)

# loading tokenizer 
tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer,
                                            cache_dir = '/efs-storage/research/tokenizer/')

# loading all the models in memory
model_path = [ args.model_dir + name for name in os.listdir(args.model_dir)]
models = [ICTrainer.load_from_checkpoint(path,args=args) for path in model_path]

testSet = test_template(args.dataset, args.test_dir)

for testset_name,testset_path in testSet.items():
    
    f1 = []
    print('calculating metrics for the testset: ', testset_name)
    for model in models:

        for test_file in testset_path:
        
            testDS = dataset(test_file)
            testDL = DataLoader( testDS, batch_size=args.batch_size, collate_fn=partial(collate_sup,tokenizer = tokenizer))

            out = trainer.test(model=model ,test_dataloaders=testDL)
            
            f1.append(out[0]['test_intent_F1'])
        
    print('test_f1: ',cal_mean_stderror(f1))