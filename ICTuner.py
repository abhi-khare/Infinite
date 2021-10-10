import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from transformers import DistilBertTokenizerFast

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl

import argparse

from transformers.models.auto.tokenization_auto import logger
from .scripts.utils import accuracy
from .scripts.dataset_scripts import dataloader
from .scripts.utils import F1

class intent_classifier(nn.Module):

    def __init__(self, args, intent_dropout, intent_hidden):

        super(intent_classifier, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
                                        args.encoder, 
                                        return_dict=True, 
                                        output_hidden_states=True,
                                        sinusoidal_pos_embds=True, 
                                        cache_dir='/efs-storage/model/'
                                    )

        self.intent_head = nn.Sequential(
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(intent_dropout),
                                        nn.Linear(768, intent_hidden),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(intent_dropout),
                                        nn.Linear(intent_hidden, args.num_class)
                                        )

        self.CE_loss = nn.CrossEntropyLoss()

        self.args = args


    def forward(self, input_ids, attention_mask, intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_logits = self.intent_head(intent_hidden)

        # accumulating intent classification loss
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        return {
            "ic_loss": intent_loss,
            "intent_pred": intent_pred
        }

class ICTrainer(pl.LightningModule):
    
    def __init__(self, args, dropout, hidden, lr):
        super().__init__()
        
        self.model = intent_classifier(args, dropout, hidden)
        self.args = args
        self.lr = lr

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
         return torch.optim.AdamW(self.parameters(), lr = self.lr , weight_decay = self.args.l2)


def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    hidden = trial.suggest_int("hidden", 32, 512, log=True)
    lr = trial.suggest_float("lr", 0.00001, 0.00006)

    model = ICTrainer(args, dropout, hidden, lr)
    dl = dataloader(args)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epoch=25,
        gpus=-1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )

    hyperparameters = dict(lr=lr, dropout=dropout, hidden=hidden)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dl)

    return trainer.callback_metrics["val_acc"].item()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" Base intent classifier HP tuner.")
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')

    # training params
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='BASELINE')
    # data params
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--intent_count', type=int)
    parser.add_argument('--slots_count',type=int)
    parser.add_argument('--dataset',type=str)

    #misc params
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--logging_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=40, timeout=60000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))
