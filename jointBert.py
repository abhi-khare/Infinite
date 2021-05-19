import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import DistilBertModel, DistilBertTokenizerFast

import pandas as pd
import random

from scripts.dataset import *
from scripts.model import IC_NER
from scripts.utils import *
from arguments import jointBert_argument


args = jointBert_argument()

if args.fix_seed:
    seed_everything(42)

if args.device == "cuda":
    gpus = -1
else:
    gpus = 0

# setting up logger
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

# fetching slot-idx dictionary
final_slots = pd.read_csv(
    "./data/multiATIS/slots_list.csv", sep=",", header=None, names=["SLOTS"]
).SLOTS.values.tolist()
idx2slots = {idx: slots for idx, slots in enumerate(final_slots)}

# setting checkpoint callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=args.weight_dir,
    monitor="val_IC_NER_loss",
    mode="min",
    filename="jointBert-{epoch:02d}-{val_loss}",
)


class jointBert(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.IC_NER = IC_NER(args)
        self.args = args

    def forward(self, input_ids, attention_mask, intent_target, slots_target):
        return self.IC_NER(input_ids, attention_mask, intent_target, slots_target)

    def training_step(self, batch, batch_idx):

        token_ids, attention_mask = batch["token_ids"], batch["mask"]
        intent_target, slots_target = batch["intent_id"], batch["slots_id"]

        out = self(token_ids, attention_mask, intent_target, slots_target)

        self.log(
            "train_IC_NER_loss",
            out["joint_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_IC_loss", out["ic_loss"], on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "train_NER_loss", out["ner_loss"], on_step=False, on_epoch=True, logger=True
        )

        return out["joint_loss"]

    def validation_step(self, batch, batch_idx):

        token_ids, attention_mask = batch["token_ids"], batch["mask"]
        intent_target, slots_target = batch["intent_id"], batch["slots_id"]

        out = self(token_ids, attention_mask, intent_target, slots_target)
        intent_pred, slot_pred = out["intent_pred"], out["slot_pred"]

        self.log(
            "val_IC_NER_loss",
            out["joint_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_IC_loss", out["ic_loss"], on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "val_NER_loss", out["ner_loss"], on_step=False, on_epoch=True, logger=True
        )
        self.log(
            "val_intent_acc",
            accuracy(out["intent_pred"], intent_target),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "slot_f1",
            slot_F1(out["slot_pred"], slots_target, idx2slots),
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return out["joint_loss"]

    def test_step(self, batch, batch_idx):

        token_ids, attention_mask = batch["token_ids"], batch["mask"]
        intent_target, slots_target = batch["intent_id"], batch["slots_id"]

        out = self(token_ids, attention_mask, intent_target, slots_target)
        intent_pred, slot_pred = out["intent_pred"], out["slot_pred"]

        self.log(
            "test_intent_acc",
            accuracy(intent_pred, intent_target),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_slot_f1",
            slot_F1(slot_pred, slots_target, idx2slots),
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return out["joint_loss"]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=3e-5, weight_decay=args.weight_decay
        )


dm = NLU_Dataset_pl(
    args.train_dir,
    args.val_dir,
    args.val_dir,
    args.tokenizer_name,
    args.max_len,
    args.batch_size,
    args.num_worker,
)

model = jointBert(args)


trainer = pl.Trainer(
    gpus=gpus,
    precision=args.precision,
    accumulate_grad_batches=args.accumulate_grad,
    max_epochs=args.epoch,
    check_val_every_n_epoch=1,
    logger=tb_logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, dm)