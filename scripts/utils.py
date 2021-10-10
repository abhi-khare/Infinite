import torch 
from torchmetrics.functional import f1

def accuracy(pred, target):
    
    return torch.sum(pred==target)/float(len(target))

def F1(pred, target):

    return f1(pred, target)
    

