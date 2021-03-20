import torch 
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score

def accuracy(pred,target):
    return torch.sum(pred==target)/len(target)

def slot_F1(pred,target):
    return f1_score( target, pred)

def label2slotType(slot_label,path):
    final_slots = pd.read_csv(path,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
    idx2slots  = {idx:slots for idx,slots in enumerate(final_slots)}
    
    slot_type = []
    for _ in slot_label:
        _ = []
        for __ in _:
            _.append(idx2slots[__])
        slot_type.append(_)
    
    return slot_type


