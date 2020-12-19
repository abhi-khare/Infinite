import os,time, pickle 
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import cuda
from models import jointBert
from dataset import nluDataset
from torch.utils.tensorboard import SummaryWriter
from seqeval.metrics import f1_score


parser = argparse.ArgumentParser()
###################################################################################################################
# model parameters
parser.add_argument('--model_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_weights', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--slots_dropout_val', type=float, default=0.1)
parser.add_argument('--intent_dropout_val', type=float, default=0.1)
parser.add_argument('--joint_loss_coef', type=float, default=1.0)
parser.add_argument('--freeze_encoder', type=bool , default=False)

#training parameters 
parser.add_argument('--encoder_lr', type=float , default=0.0005)
parser.add_argument('--rest_lr', type=float , default=0.002)
parser.add_argument('--epoch',type=int,default=25)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--check_val_every_n_epoch',type=int,default=1)
parser.add_argument('--weight_decay',type=float,default=0.003)
parser.add_argument('--shuffle_data', type=bool , default=True)
parser.add_argument('--num_worker', type=int , default=4)

# data
parser.add_argument('--train_dir',type=str)
parser.add_argument('--val_dir',type=str)
parser.add_argument('--intent_num', type=int, default=17)
parser.add_argument('--slots_num', type=int , default=160)
parser.add_argument('--max_len', type=int, default=56)

#misc. 
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_name', type=str)

args = parser.parse_args()

writer = SummaryWriter(args.exp_name)
###################################################################################################################

# loading id to slots dictionary
with open('./notebooks/map_ids_slots.pickle', 'rb') as handle:
    map_idx_slots = pickle.load(handle)

def accuracy(pred,target):
    return torch.sum(pred==target)/args.batch_size


#############################################


# instantiating a model
model = jointBert(args).to(device=args.device)

# creating train and val dataset
train_DS, val_DS =  nluDataset(args.train_dir,args.tokenizer_weights,args.max_len,args.device), nluDataset(args.val_dir,args.tokenizer_weights,args.max_len,args.device)

# train and val dataloader
train_DL = DataLoader(train_DS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)
val_DL = DataLoader(val_DS,batch_size=args.batch_size,shuffle=args.shuffle_data,num_workers=args.num_worker)

# freezing base bert model
if args.freeze_encoder:
    for params in model.encoder.parameters():
        params.requires_grad = False

# optimizer
optimizer =  optim.Adam([{'params': model.encoder.parameters(), 'lr': args.encoder_lr}], lr=args.rest_lr,weight_decay=1e-3)

# training loop
print('*'*10  + 'Training loop started' + '*'*10)
#scaler = torch.cuda.amp.GradScaler()
for _ in range(1,args.epoch):

    epoch_loss,num_batch = 0.0,0
    model.train()
    start_train = time.time()
    for idx,batch in enumerate(train_DL,0):
        num_batch += 1

        token_ids = batch['token_ids'].to(args.device, dtype = torch.long)
        mask = batch['mask'].to(args.device, dtype = torch.long)
        intent_target = batch['intent_id'].to(args.device, dtype = torch.long)
        slots_target = batch['slots_id'].to(args.device, dtype = torch.long)
        slots_label = batch['slots_label']
        slots_mask = batch['slots_mask'].to(args.device, dtype = torch.long)
        # zero the parameter gradients
        #with torch.cuda.amp.autocast():
        joint_loss ,sp,ip = model(token_ids,mask,intent_target,slots_target,slots_mask)
        
        #scaler.scale(joint_loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        joint_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += joint_loss.detach()
        
    
    epoch_loss = epoch_loss/float(num_batch)
    end_train = time.time()
    writer.add_scalar('Loss/train', epoch_loss, _)
    print("Train Epoch: {epoch_no} train_loss: {loss} time elapsed: {time}".format(epoch_no = _ , loss = epoch_loss , time = end_train - start_train))

    # validation loop
    if _% args.check_val_every_n_epoch == 0:
        print('*'*10  + 'Validation loop started' + '*'*10)
        model.eval()
        val_loss, slots_F1, intent_acc = 0,0,0
        num_batch = 0
        with torch.no_grad():
            for idx,batch in enumerate(val_DL,0):
                
                num_batch +=1
                token_ids = batch['token_ids'].to(args.device, dtype = torch.long)
                mask = batch['mask'].to(args.device, dtype = torch.long)
                intent_target = batch['intent_id'].to(args.device, dtype = torch.long)
                slots_target = batch['slots_id'].to(args.device, dtype = torch.long)
                slots_label = batch['slots_label']
                slots_mask = batch['slots_mask'].to(args.device, dtype = torch.long)

                joint_loss , slots_pred, intent_pred = model(token_ids,mask,intent_target,slots_target,slots_mask)
                slots_target,slots_pred = getSlotLabels(slots_label,slots_pred,map_idx_slots)
                
                slots_F1 += f1_score(slots_target,slots_pred)
                val_loss += joint_loss.detach()
                intent_acc += accuracy(intent_pred,intent_target)

        
        end_val = time.time()

        slots_F1  = slots_F1/float(num_batch)
        val_loss = val_loss/float(num_batch)
        intent_acc = intent_acc/float(num_batch)
        
        writer.add_scalar('Loss/val', val_loss, _ )
        writer.add_scalar('intent_acc/val', intent_acc, _ )
        writer.add_scalar('slot_F1/val', slots_F1, _ )
        
        print("Val Epoch: {epoch_no} eval_loss: {loss} intent_acc:{acc} slots_F1: {F1} time elapsed: {time}".format(epoch_no = _  , acc= intent_acc,F1=slots_F1,   loss = eval_loss , time = end_val - start_val))
        print('*'*10  + 'Training loop started' + '*'*10)

writer.close()
