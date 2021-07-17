import torch
import pandas as pd
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.scheme import IOB2


with open("./data/BG_Noise_Phrase.txt") as f:
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

def mergelistsMC(s, prob):

    anno = deepcopy(s)

    bernaulliSample = [0] * int((1000) * prob) + [1] * int(1000 * (1 - prob))
    random.shuffle(bernaulliSample)

    orig,aug  = [anno[0]],[anno[0]]
    for idx,tokens in enumerate(anno[1:]):
        
        if random.sample(bernaulliSample, 1)[0] == 0:
            orig.append([tokens[0],1000000])
            continue
        else:
            aug.append(tokens)

    return orig,aug


def contrastiveSampleGeneration(sample, noiseType):

    samplePacked = [[token, idx] for idx, token in enumerate(sample.split())]

    noisyTEXT = random.sample(phrase, 3)
    noisyTEXT = (noisyTEXT[0] + noisyTEXT[1] + noisyTEXT[2]).split(" ")
    noisyTOKENS = random.sample(noisyTEXT, random.sample([5, 6, 7,8,9,10], 1)[0])
    noisyPacked = [[token, 2000000] for idx, token in enumerate(noisyTOKENS)]

    noiseLevel = random.sample([0.25,0.50,0.75],1)[0]
    
    if noiseType == 'MC':
        orig, aug = mergelistsMC(samplePacked, prob=noiseLevel)
        augText, augSlots = zip(*aug)
        origText, origSlots = zip(*orig)
        return " ".join(list(origText))," ".join(list(augText)), origSlots,augSlots
    else:
        orig, aug  = mergelistsBG(noisyPacked,samplePacked,  prob=noiseLevel)
        augText, augSlots = zip(*aug)
        origText, origSlots = zip(*orig)
        return " ".join(list(origText))," ".join(list(augText)), origSlots,augSlots



def accuracy(pred,target):
    return torch.sum(pred==target)/float(len(target))


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

