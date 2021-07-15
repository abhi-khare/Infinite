import torch
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.scheme import IOB2

def accuracy(pred,target):
    return torch.sum(pred==target)/len(target)


def slot_F1(pred,target,id2slots):

    pred_list = pred.tolist()
    target_list = target.tolist()
    
    pred_slots , target_slots = [],[]

    for idx,sample in enumerate(target_list):
        pred_sample,target_sample = [],[]
        for jdx,slot in enumerate(sample):

            if (slot == -100 or slot==0)!=True:
                target_sample.append(id2slots[slot])
                pred_sample.append(id2slots[pred_list[idx][jdx]])

        pred_slots.append(pred_sample)
        target_slots.append(target_sample)
    
    return f1_score( target_slots, pred_slots,mode='strict', scheme=IOB2)

