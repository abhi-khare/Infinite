import torch 
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score

def accuracy(pred,target):
    return torch.sum(pred==target)/len(target)

def slot_F1(pred,target):
    return f1_score( target, pred)


