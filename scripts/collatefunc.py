import torch
from .utils import *

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
        -100 if word_id == -100 else int(slot_ids[word_id]) for word_id in word_ids
    ]

    return new_labels

def batch_tokenizer(text, slots,tokenizer):

    token_ids, mask, slots_processed = [], [], []

    for idx, sampleText in enumerate(text):
        inputs = tokenizer.encode_plus(
            sampleText,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=56,
            padding="max_length",
        )
        word_ids = inputs.word_ids()
        slots_processed.append(processSlotLabel(word_ids, slots[idx]))

        token_ids.append(inputs["input_ids"])
        mask.append(inputs["attention_mask"])

    return token_ids, mask, slots_processed


def list2Tensor(items):

    tensorItems = []
    for i in items:
        tensorItems.append(torch.tensor(i, dtype=torch.long))

    return tensorItems[0], tensorItems[1], tensorItems[2], tensorItems[3]


def collate_sup(batch,tokenizer):
    
    text,intent_id,slot_id = [],[],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
        slot_id.append(datapoint['slots_id'])
    
    # tokenization
    token_ids , mask , slot_processed = [],[],[]
    for i in range(len(text)):
        
        inputs = tokenizer.encode_plus(text[i],None,add_special_tokens=True,return_token_type_ids=False,
        truncation=True,max_length=56,padding="max_length")
        word_ids = inputs.word_ids()
        slot_processed.append(processSlotLabel(word_ids, slot_id[i].split()))
        
        token_ids.append(inputs["input_ids"])
        mask.append(inputs["attention_mask"])

    token_ids = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    intent_id = torch.tensor(intent_id, dtype=torch.long)
    slot_processed = torch.tensor(slot_processed, dtype=torch.long)
    
    
    return {'token_ids':token_ids , 'mask':mask , 'intent_id':intent_id,'slots_id':slot_processed}


def collate_CT(batch, tokenizer):

    text,intent_id,slot_id = [],[],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
        slot_id.append(datapoint['slots_id'])

    # processing batch for supervised learning
    # tokenization and packing to torch tensor
    token_ids, mask, slots_ids = batch_tokenizer(text, slot_id)
    token_ids, mask, intent_id, slots_ids = list2Tensor(
        [token_ids, mask, intent_id, slots_ids]
    )

    supBatch = {
        "token_ids": token_ids,
        "mask": mask,
        "intent_id": intent_id,
        "slots_id": slots_ids,
    }

    # processing batch for hierarchial contrastive learning

    # generating contrastive pairs
    textP1, textP2, tokenID1, tokenID2, sentID1, sentID2 = contrastivePairs(
        data["TEXT"]
    )

    # tokenization and packing for pair 1
    token_ids1, mask1, token_out1 = batch_tokenizer(textP1, tokenID1)
    token_ids1, mask1, intent_id1, token_out1 = list2Tensor(
        [token_ids1, mask1, sentID1, token_out1]
    )

    # tokenization and packing for pair 2
    token_ids2, mask2, token_out2 = batch_tokenizer(textP2, tokenID2)
    token_ids2, mask2, intent_id2, token_out2 = list2Tensor(
        [token_ids2, mask2, sentID2, token_out2]
    )

    CP1 = {
        "token_ids": token_ids1,
        "mask": mask1,
        "sentId": sentID1,
        "tokenId": token_out1,
    }

    CP2 = {
        "token_ids": token_ids2,
        "mask": mask2,
        "sentId": sentID2,
        "tokenId": token_out2,
    }

    return {"supBatch": supBatch, "HCLBatch": [CP1, CP2]}
