import torch
import pandas as pd
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.scheme import IOB2

def get_idx2slots(dataset):

    if dataset == 'SNIPS':
        slot_path = './data/SNIPS/slot_list.tsv'
    elif dataset == 'ATIS':
        slot_path = './data/ATIS/slot_list.tsv'

    # loading slot file
    slots_list = pd.read_csv(slot_path,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
    idx2slots  = {idx:slots for idx,slots in enumerate(slots_list)}
    return idx2slots

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

