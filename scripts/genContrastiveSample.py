import pandas as pd
from collections import Counter
import numpy as np
from copy import deepcopy
import random
from functools import reduce
import pickle
from sklearn.model_selection import train_test_split
random.seed(42)

with open('../data/BG_Noise_Phrase.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
phrase = [x.strip() for x in content] 


def mergelists(l1, l2 , prob):
    
    spl = [0]*int((1000)*prob) + [1]*int(1000*(1-prob))
    final = []
    while len(l1) >0 and len(l2) > 0:
        if random.sample(spl,1)[0] == 0:
            final.append(l1.pop(0))
        else:
            final.append(l2.pop(0))
    if len(l1) == 0:
        final = final + l2
    else:
        final = final + l1
    
    #print(final)
    text,slot = '',''
    for token in final:
        text += token[0] + ' '
        slot += token[1] + ' '
    #print(text,slot)
    return text,slot

def BG_Noise(samples, prob):
    
    aug_text = []
    aug_id = []

    for idx,text in enumerate(samples):
        
        bg_TEXT = random.sample(phrase,1)[0]
        
        text = mergelists(bg_TEXT.split(' '), text.split(' '),prob)
        aug_text.append(text)
        aug_id.append(idx)
    
    return aug_text,aug_id

def get_phrase_length(text):
    return text.split(" ")

def carrier_aug(samples,tau):
    
    aug_text = []
    aug_id = []

    for idx,text in enumerate(samples):

        CP_idx = get_phrase_length(text)
        CP_length = len(CP_idx)

        if CP_length <= 2:
            
            aug_text.append(text)
            aug_id.append(id)

        else:

            del_count = int(CP_length/2) if CP_length <= 5 else int(tau*CP_length)
            del_index = random.sample(list(range((CP_length))),del_count)
            
            text = ' '.join([i for j, i in enumerate(text.split(' ')) if j not in del_index])
            
            aug_id.append(idx)
            aug_text.append(text)

    return aug_text,aug_id

def contrastiveSampleGeneration(samples):

    aug_sample,aug_label = [],[]

    for tau in [0.2,0.4,0.6]:
        augmentation = carrier_aug(samples,tau)

        aug_sample += augmentation[0]
        aug_label += augmentation[1]

    for tau in [0.25,0.50,0.75]:
        augmentation = BG_Noise(samples,tau)

        aug_sample += augmentation[0]
        aug_label += augmentation[1]
    
    return aug_sample + samples ,aug_label + list(range(len(samples)))
    