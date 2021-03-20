import torch 
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score

def accuracy(pred,target):
    return torch.sum(pred==target)/len(target)

def slot_F1(pred,target,id2slots):
    
    pred_list = pred.tolist()
    target_list = target.tolist()
    
    pred_slots , target_slots = [],[]

    for idx_st,t in enumerate(target_list):
        pred_sample,target_sample = [],[]
        for idx_wt,wt in enumerate(t):

            if wt != -100:
                pred_sample.append(id2slots[wt])
                target_sample.append(id2slots[pred_list[idx_st][idx_wt]])

        pred_slots.append(pred_sample)
        target_slots.append(target_sample)
    
    
    return f1_score( target_slots, pred_slots)

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


