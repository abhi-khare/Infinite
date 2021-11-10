import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import DistilBertTokenizerFast

from .collatefunc import collate_sup

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

class adversarial_dataset(Dataset):
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
        self.aug_train = args.aug_dir
        self.val_dir = args.val_dir
        self.batch_size = args.batch_size
        self.num_worker = args.num_workers
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer,
                                            cache_dir = '/efs-storage/research/tokenizer/')
        self.experiment_type = args.experiment_type
        self.args = args

    def setup(self, stage: [str] = None):

        self.train = dataset(self.train_dir)

        self.val = dataset(self.val_dir)

        if self.experiment_type == 'BASELINE' or self.experiment_type == 'ADVERSARIAL':
            self.train_collate = partial(collate_sup,tokenizer = self.tokenizer)
            self.val_collate = partial(collate_sup, tokenizer = self.tokenizer)
        


    def train_dataloader(self):

        if self.experiment_type == 'BASELINE':
            return DataLoader( self.train, batch_size=self.batch_size, 
                                  shuffle=True, collate_fn=self.train_collate, 
                                  num_workers=self.num_worker
                             )

        elif self.experiment_type == 'ADVERSARIAL':

            orig_dl = DataLoader( self.train, batch_size=self.batch_size, 
                                  shuffle=True, collate_fn=self.train_collate, 
                                  num_workers=self.num_worker
                                )

            adv_dl = DataLoader( self.aug_train, batch_size=self.batch_size, 
                                 shuffle=True, collate_fn=self.train_collate, 
                                 num_workers=self.num_worker
                               )

            return {"orig": orig_dl, "adv": adv_dl}
        

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, collate_fn=self.val_collate, num_workers=self.num_worker
        )




