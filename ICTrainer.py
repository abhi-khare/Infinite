import pandas as pd

import torch 
import pytorch_lightning as pl
from   pytorch_lightning import seed_everything, loggers as pl_loggers
from   pytorch_lightning.callbacks import ModelCheckpoint

from scripts.dataset_scripts import dataloader
from scripts.model import intent_classifier
from scripts.utils import F1
from arguments import ICTrainer_args

args = ICTrainer_args()

if args.freeze_args:
    seed_everything(42)


# ckpt callback config for pytorch lightning
checkpoint_callback = ModelCheckpoint(
                            monitor= 'val_intent_F1', 
                            dirpath= args.param_save_dir,
                            filename= args.exp_num + '_JB-{epoch:02d}-{val_loss:.2f}',
                            save_top_k=1, mode='max',
                        )

# tensorboard logging
tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)

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
    
    def validation_step(self, batch, batch_idx):
        
        token_ids, attention_mask = batch['supBatch']['token_ids'], batch['supBatch']['mask']
        intent_target = batch['supBatch']['intent_id']
        
        out = self(token_ids,attention_mask,intent_target)
        intent_pred = out['intent_pred']
        
        self.log('val_IC_loss', out['ic_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('val_intent_F1', F1(intent_pred,intent_target), on_step=False, on_epoch=True,  logger=True)

    def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr = self.args.lr , weight_decay = self.args.l2)


# initialize the dataloader and model
dm = dataloader(args)
model = ICTrainer(args)

trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=tb_logger, callbacks=[checkpoint_callback] ,precision=args.precision,max_epochs=args.epoch, check_val_every_n_epoch=args.checkNEpoch)

# model training
trainer.fit(model, dm)