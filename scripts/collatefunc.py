import torch

def processSlotLabel(word_ids, slot_ids):

    word_ids = [-100 if word_id == None else word_id for word_id in word_ids]

    previous_word = -100

    for idx, wid in enumerate(word_ids):

        if wid == -100:
            continue

        if wid == previous_word:
            word_ids[idx] = -100

        previous_word = wid

    new_labels = [
        -100 if word_id == -100 else slot_ids[word_id] for word_id in word_ids
    ]

    return new_labels

def collate_contrast(batch, tokenizer):
    return 1


def collate_sup(batch,tokenizer):
    
    text,intent_id,slot_id = [],[],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
        slot_id.append(datapoint['slots_id'])
    
    # tokenization
    token_ids , mask , slot_out = [],[],[]
    for i in range(len(text)):
        
        inputs = tokenizer.encode_plus(text[i],None,add_special_tokens=True,return_token_type_ids=False,
        truncation=True,max_length=56,padding="max_length")
        word_ids = inputs.word_ids()
        slot_out.append(processSlotLabel(word_ids, slot_id[i]))
        
        token_ids.append(inputs["input_ids"])
        mask.append(inputs["attention_mask"])
    
    # slot_processing
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    intent_id = torch.tensor(intent_id, dtype=torch.long)
    slot_out = torch.tensor(slot_out, dtype=torch.long)
    
    
    return {'token_ids':token_ids , 'mask':mask , 'intent_id':intent_id,'slots_id':slot_out}
