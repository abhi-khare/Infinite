import torch 

def accuracy(pred,target):
    return torch.sum(pred==target)/float(len(target))

