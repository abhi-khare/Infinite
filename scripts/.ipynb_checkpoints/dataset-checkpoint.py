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
        new_labels = [idx if idx!=None else -100 for idx in word_ids]
        
        
        previous_word_idx = None
        #print(word_ids,slot_ids)
        for idx,_ in enumerate(new_labels[1:]):
            
            if _ == -100:
                continue
            if _ == previous_word_idx:
                new_labels[idx+1] = -100
            
            previous_word_idx = _
                
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
        
       # print(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"]),inputs.word_ids())
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