import torch
import random
import pandas as pd
from copy import deepcopy
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


with open("/efs-storage/Infinite/data/BG_Noise_Phrase.txt") as f:
    content = f.readlines()
phrase = [x.strip() for x in content]

# contrastive noise augmentation samples


def mergelistsBG(ns, s, prob):

    noisySample = deepcopy(ns)
    sample = deepcopy(s)

    bernaulliSample = [0] * int((1000) * prob) + [1] * int(1000 * (1 - prob))
    random.shuffle(bernaulliSample)

    final = []

    while len(noisySample) > 0 and len(sample) > 0:

        if random.sample(bernaulliSample, 1)[0] == 0:
            final.append(noisySample.pop(0))
        else:
            final.append(sample.pop(0))

    if len(noisySample) == 0:
        final = final + sample
    else:
        final = final + noisySample

    return s,final

def mergelistsMC(text_packed, prob):

    text = deepcopy(text_packed)

    bernaulliSample = [0] * int((1000) * prob) + [1] * int(1000 * (1 - prob))
    random.shuffle(bernaulliSample)

    orig,aug  = [text[0]],[text[0]]
    for idx,tokens in enumerate(text[1:]):
        
        if random.sample(bernaulliSample, 1)[0] == 0:
            orig.append([tokens[0],'2000'])
        else:
            orig.append(tokens)
            aug.append(tokens)

    return orig,aug

def contrastiveSampleGenerator(sample, noise_type):

    samplePacked = [[token, str(idx)] for idx, token in enumerate(sample.split())]

    noisyTEXT = random.sample(phrase, 3)
    noisyTEXT = (noisyTEXT[0] + noisyTEXT[1] + noisyTEXT[2]).split()
    noisyTOKENS = random.sample(noisyTEXT, random.sample([5, 6, 7,8,9,10], 1)[0])
    noisyPacked = [[token, '2000'] for idx, token in enumerate(noisyTOKENS)]

    if noise_type == 'MC':
        noise_param = random.sample([0.20,0.40,0.60],1)[0]
        orig, aug = mergelistsMC(samplePacked, prob=noise_param)
        augText, augSlots = zip(*aug)
        origText, origSlots = zip(*orig)

        return ' '.join(list(origText)), ' '.join(list(augText)), ' '.join(list(origSlots)), ' '.join(list(augSlots))

    elif noise_type == 'BG':
        noise_param = random.sample([0.20,0.40,0.60],1)[0]
        orig, aug  = mergelistsBG(noisyPacked,samplePacked,  prob=noise_param)
        augText, augSlots = zip(*aug)
        origText, origSlots = zip(*orig)
        return ' '.join(list(origText)), ' '.join(list(augText)), ' '.join(list(origSlots)), ' '.join(list(augSlots))

def contrastivePairs(text, noise_type):

    textP1, textP2, slotsID1, slotsID2, sentID1, sentID2 = [], [], [], [], [], []

    for idx, sample in enumerate(text):

        origText,augText, origSlots, augSlots = contrastiveSampleGenerator(sample,noise_type)
          
        textP1.append(origText)
        slotsID1.append(origSlots)
        sentID1.append(idx)

        textP2.append(augText)
        slotsID2.append(augSlots)
        sentID2.append(idx)

    return textP1, textP2, slotsID1, slotsID2, sentID1, sentID2

def accuracy(pred,target):
    return torch.sum(pred==target)/float(len(target))

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

