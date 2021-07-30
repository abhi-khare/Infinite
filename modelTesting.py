import os 
import pandas as pd
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from jointBertTrainer import jointBertTrainer
from scripts.dataset import dataset
from scripts.utils import accuracy, slot_F1, cal_mean_stderror
from scripts.collatefunc import collate_sup
from data.testTemplate import test_template



parser = argparse.ArgumentParser()
# model params
parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
parser.add_argument("--intent_hidden", type=int)
parser.add_argument("--slots_hidden", type=int)
parser.add_argument("--intent_dropout", type=float)
parser.add_argument("--slots_dropout", type=float)
parser.add_argument('--jointCoef', type=float, default=0.50)

# training params
parser.add_argument('--lr', type=float)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--l2', type=float, default=0.003)
parser.add_argument('--mode', type=str, default='BASELINE')
parser.add_argument('--checkNEpoch', type=int, default=1)

# data params
parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)
parser.add_argument('--intent_count', type=int)
parser.add_argument('--slots_count',type=int)
parser.add_argument('--dataset',type=str)

#misc params
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--param_save_dir', type=str)
parser.add_argument('--logging_dir', type=str)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

seed_everything(42, workers=True)

def get_idx2slots(dataset):

    if dataset == 'SNIPS':
        slot_path = '/efs-storage/Infinite/data/SNIPS/slots_list.tsv'
    elif dataset == 'ATIS':
        slot_path = '/efs-storage/Infinite/data/ATIS/slots_list.tsv'

    # loading slot file
    slots_list = pd.read_csv(slot_path,sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
    idx2slots  = {idx:slots for idx,slots in enumerate(slots_list)}
    return idx2slots

idx2slots = get_idx2slots(args.dataset)

trainer = pl.Trainer(gpus=args.gpus, deterministic=True,precision=args.precision)

testSet = test_template(args.dataset)

# testing the model
          
model_path = [ args.base_dir + name for name in os.listdir(args.base_dir)]

for path in model_path:

    model = jointBertTrainer.load_from_checkpoint(path,args=args)

    for testName,test in testSet.items():

        acc,slotF1 = [],[]
        print('calculating metrics for the testset: ', testName)

        for test_fn in test:
        
            testDS = dataset(test_fn)
            testDL = DataLoader( testDS, batch_size=1, collate_fn=collate_sup)

            out = trainer.test(model=model ,test_dataloaders=testDL)
            
            acc.append(out[0]['test_intent_acc'])
            slotF1.append(out[0]['test_slot_f1'])
        
        print('acc: ',cal_mean_stderror(acc),'slotsF1: ',cal_mean_stderror(slotF1))