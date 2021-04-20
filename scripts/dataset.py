import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
import pandas as pd
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class nluDataset(Dataset):

    def __init__(self, file_dir, tokenizer, max_len):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.max_len = max_len
    
    def processSlotLabel(self,word_ids,slot_ids,text):
        
        slot_ids = list(map(int, slot_ids.split(' ')))
        
        new_labels = [-100 if idx==None else slot_ids[idx] for idx in word_ids]
        previous_word_idx = -100
        for idx,_ in enumerate(new_labels[1:]):
        
            if _ == -100:
                continue
            if _ == previous_word_idx:
                new_labels[idx+1] = -100
        
        
            previous_word_idx = _
            
        return new_labels
        

    def __getitem__(self, index):
        
        text = str(self.data.TEXT[index])
        text = text.replace('.','')
        text = text.replace('\'','')
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
        )
        
        # text encoding
        token_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        word_ids = inputs.word_ids()

        # intent
        intent_id = torch.tensor(self.data.INTENT_ID[index], dtype=torch.long)
        intent_label = self.data.INTENT[index]

        # label processing
        slot_label = self.data.SLOTS[index]
        slot_id = self.processSlotLabel(word_ids,self.data.SLOTS_ID[index],text)
    
        slot_id = torch.tensor(slot_id,dtype=torch.long)
        
        return {
            'token_ids': token_ids,
            'mask': mask,
            'intent_id': intent_id,
            'slots_id' : slot_id,
            'intent_label': intent_label,
            'slots_label' : slot_label
        } 
    
    def __len__(self):
        return len(self.data)


class NLU_Dataset_pl(pl.LightningDataModule):
    
    def __init__(self, train_dir, val_dir, test_dir,tokenizer, max_len, batch_size,num_worker):
        
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_worker = num_worker

    def setup(self,stage: [str] = None): 
        self.train = nluDataset( self.train_dir, self.tokenizer, self.max_len)
        
        self.val = nluDataset( self.val_dir, self.tokenizer, self.max_len)
        
        self.test =  nluDataset( self.test_dir, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,num_workers=self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,num_workers=self.num_worker)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,num_workers=self.num_worker)

