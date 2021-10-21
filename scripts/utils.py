import torch 
from torchmetrics.functional import f1

def accuracy(pred, target):
    
    return torch.sum(pred==target)/float(len(target))

def F1(pred, target):

    return f1(pred, target)

def cal_mean_stderror(metric):
    
    if len(metric) == 1:
        return metric
    var,std_error = 0,0
    mean = sum(metric)/len(metric)
    for m in metric:
        var += (m-mean)**2
    var = (var/(len(metric)-1))**0.5
    std_error = var/((len(metric))**0.5)
    return [round(mean,4),round(std_error,4)]
    

