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
