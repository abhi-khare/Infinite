import pandas as pd

def get_carrier_phrase_length(annotations):
    CP_idx = []
    for idx,token in enumerate(annotations.split(' ')):
        if token == 'O':
            CP_idx.append(idx)
    
    return CP_idx

def carrier_aug(data,tau):
    
    orig_data = deepcopy(data)
    
    augINTENT, augSLOTS, augTEXT,augID = [],[],[],[]
    
    cnt = 0
    for sample in data.values.tolist():
        
        
        CP_idx = get_carrier_phrase_length(sample[2])
        
        CP_length = len(CP_idx)
        
        if CP_length <= 1:
            del_count = CP_length
        else:
            del_count = 1 if CP_length <=3 else int(tau*CP_length)
        
        del_index = random.sample(CP_idx,del_count)

        augINTENT.append(sample[3])
        TEXT = ' '.join([i for j, i in enumerate(sample[1].split(' ')) if j not in del_index])
        SLOTS = ' '.join([i for j, i in enumerate(sample[2].split(' ')) if j not in del_index])
        augTEXT.append(TEXT)
        augSLOTS.append(SLOTS)
        augID.append(cnt)
        cnt+=1
    
               
    augPD = pd.DataFrame([augID,augTEXT,augSLOTS,augINTENT],index=['ID','TEXT','SLOTS','INTENT']).T
    
    return augPD