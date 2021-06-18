import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from transformers import DistilBertTokenizerFast

import pandas as pd
from copy import deepcopy
import random

basePath = "./"
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

# loading background noise samples
with open(basePath + "data/BG_Noise_Phrase.txt") as f:
    content = f.readlines()
phrase = [x.strip() for x in content]

# contrastive noise augmentation samples


def mergelists(cs, s, prob):

    contraSample = deepcopy(cs)
    sample = deepcopy(s)

    bernaulliSample = [0] * int((1000) * prob) + [1] * int(1000 * (1 - prob))
    random.shuffle(bernaulliSample)

    final = []

    while len(contraSample) > 0 and len(sample) > 0:

        if random.sample(bernaulliSample, 1)[0] == 0:
            final.append(contraSample.pop(0))
        else:
            final.append(sample.pop(0))

    if len(contraSample) == 0:
        final = final + sample
    else:
        final = final + contraSample

    return final


def carrier_aug(sample, tau):

    CP_length = len(sample)

    if CP_length <= 2:
        return sample
    else:

        del_count = int(CP_length / 2) if CP_length <= 5 else int(tau * CP_length)
        del_index = random.sample(list(range((CP_length))), del_count)

        sampled_tokens = [
            token for jdx, token in enumerate(sample) if jdx not in del_index
        ]
        return sampled_tokens


def contrastiveSampleGeneration(sample, slots):

    contraSample, contraTokenLabel = [], []
    samplePacked = [[token, slots[idx]] for idx, token in enumerate(sample.split())]

    noisyTEXT = random.sample(phrase, 3)
    noisyTEXT = (noisyTEXT[0] + noisyTEXT[1] + noisyTEXT[2]).split(" ")
    noisyTOKENS = random.sample(noisyTEXT, random.sample([3, 4, 5, 6, 7], 1)[0])
    noisyPacked = [[token, -100] for idx, token in enumerate(noisyTOKENS)]

    cAug1 = carrier_aug(samplePacked, tau=0.20)
    cAug2 = carrier_aug(samplePacked, tau=0.40)
    cAug3 = carrier_aug(samplePacked, tau=0.30)
    BGAug1 = mergelists(samplePacked, noisyPacked, prob=0.25)
    BGAug2 = mergelists(samplePacked, noisyPacked, prob=0.50)
    BGAug3 = mergelists(samplePacked, noisyPacked, prob=0.75)

    for sample in [cAug1, cAug2, cAug3, BGAug1, BGAug2, BGAug3]:
        text, slots = zip(*sample)
        contraSample.append(" ".join(list(text)))
        contraTokenLabel.append(list(slots))

    return contraSample, contraTokenLabel


def processSlotLabel(word_ids, slot_ids):

    # replace None and repetition with -100

    word_ids = [-100 if word_id == None else word_id for word_id in word_ids]

    previous_word = -100

    for idx, wid in enumerate(word_ids):

        if wid == -100:
            continue

        if wid == previous_word:
            word_ids[idx] = -100

        previous_word = wid

    # slot_ids = list(map(int, slot_ids.split(" ")))
    new_labels = [
        -100 if word_id == -100 else slot_ids[word_id] for word_id in word_ids
    ]

    return new_labels


def batch_tokenizer(text, slotsID):
    token_ids, mask, slot_out = [], [], []
    for idx, sampleText in enumerate(text):
        inputs = tokenizer.encode_plus(
            sampleText,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=56,
            padding="max_length",
        )
        word_ids = inputs.word_ids()
        slot_out.append(processSlotLabel(word_ids, slotsID[idx]))

        token_ids.append(inputs["input_ids"])
        mask.append(inputs["attention_mask"])

    return token_ids, mask, slot_out


def list2Tensor(items):
    tensorItems = []
    for i in items:
        try:
            tensorItems.append(torch.tensor(i, dtype=torch.long))
        except:
            a = 1

    return tensorItems[0], tensorItems[1], tensorItems[2], tensorItems[3]


def contrastivePairs(text):

    # generating global sampleID and tokenID

    tokenID, tokCnt = [], 0

    for sample in text:
        sampleLen = len(sample.split())
        tokenID.append(list(range(tokCnt, tokCnt + sampleLen)))
        tokCnt += sampleLen

    textP1, textP2, slotsID1, slotsID2, sentID1, sentID2 = [], [], [], [], [], []

    for idx, sample in enumerate(text):
        # processing pair 1
        textP1.append(sample)
        slotsID1.append(tokenID[idx])
        sentID1.append(idx)

        # processing pair 2
        augText, augSlots = contrastiveSampleGeneration(sample, tokenID[idx])
        augID = random.sample([0, 2, 3, 4, 5, 1], 1)[0]

        textP2.append(augText[augID])
        slotsID2.append(augSlots[augID])
        sentID2.append(idx)

    return textP1, textP2, slotsID1, slotsID2, sentID1, sentID2


def train_collate(batch):

    data = {key: [] for key in batch[0].keys()}
    temp = {data[key].append(sample[key]) for sample in batch for key in sample.keys()}

    # processing batch for supervised learning
    # tokenization and packing to torch tensor
    token_ids, mask, slots_out = batch_tokenizer(data["TEXT"], data["slotsID"])
    token_ids, mask, intent_id, slots_out = list2Tensor(
        [token_ids, mask, data["intentID"], slots_out]
    )

    # processing batch for contrastive learning
    # tokenization and packing to torch tensor

    supBatch = {
        "token_ids": token_ids,
        "mask": mask,
        "intent_id": intent_id,
        "slots_id": slots_out,
    }

    # processing batch for hierarchial contrastive learning

    # generating contrastive pairs
    textP1, textP2, slotsID1, slotsID2, sentID1, sentID2 = contrastivePairs(
        data["TEXT"]
    )
    print(textP1, textP2, slotsID1, slotsID2, sentID1, sentID2)

    # tokenization and packing for pair 1
    token_ids1, mask1, slots_out1 = batch_tokenizer(textP1, slotsID1)
    token_ids1, mask1, intent_id1, slots_out1 = list2Tensor(
        [token_ids1, mask1, sentID1, slots_out1]
    )
    # tokenization and packing for pair 2
    token_ids2, mask2, slots_out2 = batch_tokenizer(textP2, slotsID2)
    token_ids2, mask2, intent_id2, slots_out2 = list2Tensor(
        [token_ids2, mask2, sentID2, slots_out2]
    )

    CP1 = {
        "token_ids": token_ids1,
        "mask": mask1,
        "intent_id": intent_id1,
        "slots_id": slots_out1,
    }

    CP2 = {
        "token_ids": token_ids2,
        "mask": mask2,
        "intent_id": intent_id2,
        "slots_id": slots_out2,
    }

    return {"supBatch": supBatch, "HCLBatch": [CP1, CP2]}


def val_collate(batch):

    data = {key: [] for key in batch[0].keys()}
    temp = {data[key].append(sample[key]) for sample in batch for key in sample.keys()}

    # processing batch for supervised learning
    # tokenization and packing to torch tensor
    token_ids, mask, slots_out = batch_tokenizer(data["TEXT"], data["slotsID"])
    token_ids, mask, intent_id, slots_out = list2Tensor(
        [token_ids, mask, data["intentID"], slots_out]
    )

    # processing batch for contrastive learning
    # tokenization and packing to torch tensor

    supBatch = {
        "token_ids": token_ids,
        "mask": mask,
        "intent_id": intent_id,
        "slots_id": slots_out,
    }
    return {"supBatch": supBatch}


class HCLDataset(Dataset):
    def __init__(self, file_dir):

        self.data = pd.read_csv(file_dir, sep="\t", header=0)

    def process_text(self, text):
        text = text.replace(".", "")
        text = text.replace("'", "")
        text = " ".join(text.split())
        return text

    def __getitem__(self, index):

        text = str(self.data.TEXT[index])
        text = self.process_text(text)

        sampleID = self.data.ID[index]

        intentID = self.data.INTENT_ID[index]

        slotsID = [int(sid) for sid in self.data.SLOTS_ID[index].split(" ")]

        return {
            "TEXT": text,
            "sampleID": sampleID,
            "intentID": intentID,
            "slotsID": slotsID,
        }

    def __len__(self):
        return len(self.data)


class HCLDataset_pl(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size, num_worker):

        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_worker = num_worker

    def setup(self, stage: [str] = None):
        self.train = HCLDataset(self.train_dir)

        self.val = HCLDataset(self.val_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=train_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=18, num_workers=self.num_worker, collate_fn=val_collate
        )
