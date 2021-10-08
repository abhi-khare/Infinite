import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import AutoTokenizer
from .collatefunc import collate_sup, collate_AT, collate_CT, collate_CT_AT

class dataset(Dataset):
    def __init__(self, file_dir: str) -> None:

        self.data = pd.read_csv(file_dir, sep="\t")

    def __getitem__(self, index: int) -> dict:
        
        # text
        text = str(self.data.TEXT[index])
        
        # intent
        intent_label = self.data.INTENT[index]
        intent_id = self.data.INTENT_ID[index]

        return {

            "text": text,
            "intent_id": intent_id,
            "intent_label": intent_label
        }

    def __len__(self):
        return len(self.data)


class dataloader(pl.LightningDataModule):
    
    def __init__( self, args):

        super().__init__()

        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.batch_size = args.batch_size
        self.num_worker = args.num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                            cache_dir = '/efs-storage/tokenizer/')
        self.mode = args.mode
        self.args = args

    def setup(self):

        self.train = dataset(self.train_dir)

        self.val = dataset(self.val_dir)

        if self.mode == 'BASELINE':
            self.train_collate = partial(collate_sup,tokenizer = self.tokenizer)
            self.val_collate = partial(collate_sup, tokenizer = self.tokenizer)


    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_collate, num_workers=self.num_worker
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, collate_fn=self.val_collate, num_workers=self.num_worker
        )




