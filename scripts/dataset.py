import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
import pandas as pd
import random

class nluDataset(Dataset):

    def __init__(self, file_dir, tokenizer, max_len, device):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.max_len = max_len
    
    def processSlotLabel(self,word_ids,slot_ids):
        
        slot_ids = list(map(int, slot_ids.split(' ')))   
        new_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append(-100)
            elif word_idx != previous_word_idx:
                new_labels.append(slot_ids[word_idx])
            else:
                new_labels.append(-100)
        
        return new_labels  
        

    def __getitem__(self, index):
        
        text = str(self.data.TEXT[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            
            #is_split_into_words=True
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
        slot_id = self.processSlotLabel(word_ids,self.data.SLOTS_ID[index])
        slot_id = torch.tensor(slot_id,dtype=torch.long)
        #language = self.data.language[index]
        
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