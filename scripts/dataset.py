import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
import pandas as pd
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class nluDataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):

        self.data = pd.read_csv(file_dir, sep="\t")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.max_len = max_len

    def processSlotLabel(self, word_ids, slot_ids, text):

        # replace None and repetition with -100

        word_ids = [-100 if word_id == None else word_id for word_id in word_ids]

        previous_word = -100

        for idx, wid in enumerate(word_ids):

            if wid == -100:
                continue

            if wid == previous_word:
                word_ids[idx] = -100

            previous_word = wid

        slot_ids = list(map(int, slot_ids.split(" ")))
        new_labels = [
            -100 if word_id == -100 else slot_ids[word_id] for word_id in word_ids
        ]

        return new_labels

    def __getitem__(self, index):

        text = str(self.data.TEXT[index])
        text = text.replace(".", "")
        text = text.replace("'", "")
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            # is_split_into_words=True
        )

        # print(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"]),inputs.word_ids())
        # text encoding
        token_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        word_ids = inputs.word_ids()

        # intent
        intent_id = torch.tensor(self.data.INTENT_ID[index], dtype=torch.long)
        intent_label = self.data.INTENT[index]

        # label processing
        slot_label = self.data.SLOTS[index]
        slot_id = self.processSlotLabel(word_ids, self.data.SLOTS_ID[index], text)

        slot_id = torch.tensor(slot_id, dtype=torch.long)

        # language = self.data.language[index]

        return {
            "token_ids": token_ids,
            "mask": mask,
            "intent_id": intent_id,
            "slots_id": slot_id,
            "intent_label": intent_label,
            "slots_label": slot_label,
            "text": text,
            "slotsID": self.data.SLOTS_ID[index],
        }

    def __len__(self):
        return len(self.data)


class contraDataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):

        self.data = pd.read_csv(file_dir, sep="\t")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
        self.max_len = max_len

    def processSlotLabel(self, word_ids, slot_ids):

        # replace None and repetition with -100

        word_ids = [-100 if word_id == None else word_id for word_id in word_ids]

        previous_word = -100

        for idx, wid in enumerate(word_ids):

            if wid == -100:
                continue

            if wid == previous_word:
                word_ids[idx] = -100

            previous_word = wid

        slot_ids = list(map(int, slot_ids.split(" ")))
        new_labels = [
            -100 if word_id == -100 else slot_ids[word_id] for word_id in word_ids
        ]

        return new_labels

    def __getitem__(self, index):

        text = str(self.data.TEXT[index])
        text = text.replace(".", "")
        text = text.replace("'", "")
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            # is_split_into_words=True
        )

        # print(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"]),inputs.word_ids())
        # text encoding
        token_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        word_ids = inputs.word_ids()

        # intent
        slot_id = self.processSlotLabel(word_ids, self.data.tokenID[index])
        slot_id = torch.tensor(slot_id, dtype=torch.long)

        # label processing
        sent_id = self.data.ID[index]
        sent_id = torch.tensor(int(sent_id), dtype=torch.long)

        return {
            "token_ids": token_ids,
            "mask": mask,
            "sent_id": sent_id,
            "slot_id": slot_id,
        }

    def __len__(self):
        return len(self.data)


class NLU_Dataset_pl(pl.LightningDataModule):
    def __init__(
        self, train_dir, val_dir, test_dir, tokenizer, max_len, batch_size, num_worker
    ):

        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_worker = num_worker

    def setup(self, stage: [str] = None):
        self.train = nluDataset(self.train_dir, self.tokenizer, self.max_len)

        self.val = nluDataset(self.val_dir, self.tokenizer, self.max_len)

        self.test = nluDataset(self.test_dir, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_worker
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_worker
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_worker
        )


class contra_pl(pl.LightningDataModule):
    def __init__(self, train_dir, tokenizer, max_len, batch_size, num_worker):

        super().__init__()
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len

    def setup(self, stage: [str] = None):
        self.train = contraDataset(self.train_dir, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
