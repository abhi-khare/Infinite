import torch
import random
from .utils import *
from .noise_models import *

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
            sampleText.split(),
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=56,
            padding="max_length",
            is_split_into_words=True
        )
        word_ids = inputs.word_ids()
        slots_processed.append(processSlotLabel(word_ids, slots[idx].split()))

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
    token_ids, mask, slot_processed = batch_tokenizer(text, slot_id,tokenizer)

    token_ids, mask, intent_id, slot_processed = list2Tensor([token_ids, mask, intent_id, slot_processed])
    
    return {'token_ids':token_ids , 'mask':mask , 'intent_id':intent_id,'slots_id':slot_processed}

def collate_AT(batch,tokenizer,noise_type):

    text,intent_id,slot_id = [],[],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
        slot_id.append(datapoint['slots_id'])
    
    # sampling adversarial examples
    batch_size = len(text)
    idx = random.sample(list(range(batch_size)), int(0.30*batch_size))

    text = [sample for id,sample in enumerate(text) if id in idx]
    intent_id = [sample for id,sample in enumerate(intent_id) if id in idx]
    slot_id = [sample for id,sample in enumerate(slot_id) if id in idx]
    
    
    # adversarial examples
    if noise_type == 'MC':
        noise_param = random.sample([0.20,0.40,0.60],1)[0]
        adv_text,adv_intent,adv_slot = MC_noise(text,intent_id,slot_id,noise_param)
    elif noise_type == 'BG':
        noise_param = random.sample([0.25,0.50,0.75],1)[0]
        adv_text,adv_intent,adv_slot = BG_noise(text,intent_id,slot_id,noise_param)
    
    text = text + adv_text
    intent_id = intent_id + adv_intent
    slot_id = slot_id + adv_slot

    # tokenization
    token_ids, mask, slot_processed = batch_tokenizer(text, slot_id,tokenizer)
    # slot_processing
    token_ids, mask, intent_id, slot_processed = list2Tensor([token_ids, mask, intent_id, slot_processed])

    return {'token_ids':token_ids , 'mask':mask , 'intent_id':intent_id,'slots_id':slot_processed}


def collate_CT(batch, tokenizer, noise_type):

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
        text,noise_type
    )

    # tokenization and packing for pair 1
    token_ids1, mask1, processed_tokenID1 = batch_tokenizer(textP1, tokenID1)
    token_ids1, mask1, sentID1, packed_tokenID1 = list2Tensor(
        [token_ids1, mask1, sentID1, processed_tokenID1]
    )

    # tokenization and packing for pair 2
    token_ids2, mask2, processed_tokenID2 = batch_tokenizer(textP2, tokenID2)
    token_ids2, mask2, sentID2, packed_tokenID2 = list2Tensor(
        [token_ids2, mask2, sentID2, processed_tokenID2]
    )

    CP1 = {
        "token_ids": token_ids1,
        "mask": mask1,
        "sent_id": sentID1,
        "token_id": packed_tokenID1,
    }

    CP2 = {
        "token_ids": token_ids2,
        "mask": mask2,
        "sent_id": sentID2,
        "token_id": packed_tokenID2,
    }

    return {"supBatch": supBatch, "HCLBatch": [CP1, CP2]}

