import torch
import random

def batch_tokenizer(text, tokenizer):

    token_ids, mask = [], []

    for _ , sampleText in enumerate(text):
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

        token_ids.append(inputs["input_ids"])
        mask.append(inputs["attention_mask"])

    return token_ids, mask


def list2Tensor(items):

    tensorItems = []
    for i in items:
        tensorItems.append(torch.tensor(i, dtype=torch.long))

    return tensorItems[0], tensorItems[1], tensorItems[2]


def collate_sup(batch,tokenizer):
    
    text,intent_id = [],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
    
    # tokenization    
    token_ids, mask = batch_tokenizer(text, tokenizer)
    token_ids, mask, intent_id = list2Tensor([token_ids, mask, intent_id])
    
    supBatch = {
        "token_ids": token_ids,
        "mask": mask,
        "intent_id": intent_id
    }

    return {"supBatch": supBatch}

def collate_AT(batch,tokenizer,noise_type):

    text,intent_id = [],[]
    
    for datapoint in batch:
        text.append(datapoint['text'])
        intent_id.append(datapoint['intent_id'])
    
    # sampling adversarial examples
    batch_size = len(text)
    idx = random.sample(list(range(batch_size)), int(0.30*batch_size))

    text = [sample for id,sample in enumerate(text) if id in idx]
    intent_id = [sample for id,sample in enumerate(intent_id) if id in idx]
    
    # adversarial examples
    #if noise_type == 'MC':
    #    noise_param = random.sample([0.20,0.40,0.60],1)[0]
    #    adv_text,adv_intent,adv_slot = MC_noise(text,intent_id,slot_id,noise_param)
    #elif noise_type == 'BG':
    #    noise_param = random.sample([0.25,0.50,0.75],1)[0]
    #    adv_text,adv_intent,adv_slot = BG_noise(text,intent_id,slot_id,noise_param)
    
    text = text + adv_text
    intent_id = intent_id + adv_intent

    # tokenization    
    token_ids, mask = batch_tokenizer(text, tokenizer)
    token_ids, mask, intent_id = list2Tensor([token_ids, mask, intent_id])

    Batch = {
        "token_ids": token_ids,
        "mask": mask,
        "intent_id": intent_id,
    }

    return {"supBatch": supBatch}
