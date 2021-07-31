import random 
import re 

with open( '/efs-storage/Infinite/data/BG_Noise_Phrase.txt') as f:
    content = f.readlines()
noise_phrase = [x.strip() for x in content]


def merge_text_label(text,slot):
    data = []
    for i in range(len(text)):
        data.append([text[i] , slot[i]])
    return data

def twolists(l1, l2 , prob):
    
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

    text,slot = zip(*final)
    return ' '.join(list(text)), ' '.join(list(slot))

def BG_noise(text,intent,slot, prob):
    
    augINTENT, augSLOTS, augTEXT = [],[],[]
    
    for i in range(len(text)):
        
        bg_TEXT = random.sample(noise_phrase,1)[0]
        label_str = '0 '*len(bg_TEXT.split())
        bg_SLOTS = ' '.join(label_str.split())
        
        noisyData = merge_text_label(bg_TEXT.split(),bg_SLOTS.split())
        cleanData = merge_text_label(text[i].split(),slot[i].split())
        
        augText , augSlots = twolists(noisyData,cleanData,prob)
        
        augINTENT.append(intent[i])
        augTEXT.append(augText)
        augSLOTS.append(augSlots)

    return augTEXT, augINTENT, augSLOTS

def MC_noise(text,intent,slot,tau):
      
    augINTENT, augSLOTS, augTEXT,augID = [],[],[],[]
    
    for i in range(len(text)):
        
        phrase_idx = list(range(len(slot[i].split())))
        phrase_length = len(phrase_idx)
        
        if phrase_length <= 2:
            augINTENT.append(intent[i])
            augTEXT.append(text[i])
            augSLOTS.append(slot[i])
        else:
            del_count = int(phrase_length/2) if phrase_length <= 5 else int(tau*phrase_length)
            del_index = random.sample(phrase_idx,del_count)

            TEXT = ' '.join([i for j, i in enumerate(text[i].split()) if j not in del_index])
            SLOTS = ' '.join([i for j, i in enumerate(slot[i].split()) if j not in del_index])
         
            augINTENT.append(intent[i])
            augTEXT.append(TEXT)
            augSLOTS.append(SLOTS)
               
    return augTEXT, augINTENT, augSLOTS
