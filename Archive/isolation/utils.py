import torch 
import numpy
import os, stat
from os import chmod
import random

# fn to convert slots ids to slots labels
def getSlotsLabels(slots_labels,slots_pred,map_ids_slots):
    
    labels = []
    for sl in slots_labels:
        labels.append(sl.split())
    
    preds = []
    for slotSeq in slots_pred:
        slots_tokens = [ map_ids_slots[slot_id] for slot_id in slotSeq]
        preds.append(slots_tokens)

    return labels,preds



