import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd

def processLabel(labels, max_len):
    slot_label,slot_mask,slot_length = [] , [],0
    
    for sLabel in labels.split():
        slot_label.append(int(sLabel))
        slot_mask.append(1)
        slot_length +=1
    slot_label += [82]*(max_len - slot_length)
    slot_mask += [0]*(max_len - slot_length)
    
    slot_label = torch.LongTensor(slot_label)
    slot_mask = torch.LongTensor(slot_mask)
    
    return slot_label, slot_mask

class contraDataset(Dataset):

    def __init__(self, file_dir, tokenizer, max_len, device):
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
    
    def tokenize(self,text, tokenizer):
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length= self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        token_ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return token_ids, mask


    def __getitem__(self,index):

        text = str(self.data.utterance[index])
        text = " ".join(text.split())

        aug_text = str(self.data.utterance[index])
        aug_text = " ".join(aug_text.split())

        text_token_ids, text_mask = self.tokenize(text, self.tokenizer)
        aug_token_ids, aug_mask = self.tokenize(aug_text, self.tokenizer)


        return {
            'text_token_ids': torch.tensor(text_token_ids, dtype=torch.long),
            'text_mask': torch.tensor(text_mask, dtype=torch.long),

            'aug_token_ids': torch.tensor(aug_token_ids, dtype=torch.long),
            'aug_mask': torch.tensor(aug_mask, dtype=torch.long),
        } 
    
    def __len__(self):
        return len(self.data)




class nluDataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len, device):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len

    def __getitem__(self, index):
        
        text = str(self.data.utterance[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        # text id
        token_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # intent
        intent_id = torch.tensor(self.data.intent_ID[index], dtype=torch.long)
        intent_label = self.data.intent[index]
        # slot
        slot_id,slot_mask = processLabel(self.data.slots_ID[index],self.max_len)
        slot_label = self.data.slot_labels[index]

        language = self.data.language[index]
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'intent_id': intent_id,
            'slots_id' : slot_id,
            'slots_mask' : slot_mask,
            'language' : language,
            'intent_label': intent_label,
            'slots_label' : slot_label
        } 
    
    def __len__(self):
        return len(self.data)
