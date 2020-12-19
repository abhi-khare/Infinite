import time,os 
import torch 
import torch.nn as nn
from torch import cuda
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel ,DistilBertTokenizer
from pytorch_metric_learning import losses
import argparse
import optuna

def process_label(labels, max_len):
    slot_target = []
        
    for sLabel in labels:
        slots = [int(L) for L in sLabel.split()]
        slots += [159]*(max_len - len(slots))
        slot_target.append(slots)
        
    slot_target = torch.LongTensor(slot_target)
    return slot_target

class Bertencoder(nn.Module):

    def __init__(self,args):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(args.model_name,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 768)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden

class nlu_dataset(Dataset):
    def __init__(self, file_dir, tokenizer, max_len):
        
        self.data = pd.read_csv(file_dir, sep='\t')
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
    def __getitem__(self, index):
        
        text = str(self.data.utterance[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'intent': self.data.intent[index],
            'slot' : self.data.slot_labels[index],
            'intent_target': torch.tensor(self.data.intent_ID[index], dtype=torch.long),
            'slot_target' : self.data.slots_ID[index],
            'lang' : self.data.Language[index]
        } 
    
    def __len__(self):
        return len(self.data)


# command line args

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')

parser.add_argument('--train_dir', type=str)
parser.add_argument('--val_dir', type=str)

parser.add_argument('--max_len',type=int,default=46)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--exp_name',type=str)

args = parser.parse_args()


trainDS, valDS =  nlu_dataset(args.train_dir,args.tokenizer_name,args.max_len), nlu_dataset(args.val_dir,args.tokenizer_name,args.max_len)
loss_func = losses.TripletMarginLoss()

def objective(trial):
    # instantiate model
    model = Bertencoder(args).to(device=args.device)
    
    #instantiate optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) # trial for optimizer lr 
    optimizer =  optim.Adam( model.parameters() , lr=lr, weight_decay=1e-3)

    # get train and val dataloader
    trainDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=1)
    valDL = DataLoader(trainDS,batch_size=args.batch_size,shuffle=True,num_workers=1)

    # training of model
    for _ in range(1,args.epoch):
        
        model.train()
        for idx,batch in enumerate(trainDL,0):
            
            ids = batch['ids'].to(args.device, dtype = torch.long)
            mask = batch['mask'].to(args.device, dtype = torch.long)
            intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()
            embedding = model(ids,mask)
            loss = loss_func(embedding,intent_target)
            loss.backward()
            optimizer.step()



        # validation of model
    
        val_loss,num_batch = 0,0
        model.eval()
        with torch.no_grad():
            for idx,batch in enumerate(valDL,0):
                num_batch += 1
                ids = batch['ids'].to(args.device, dtype = torch.long)
                mask = batch['mask'].to(args.device, dtype = torch.long)
                intent_target = batch['intent_target'].to(args.device, dtype = torch.long)
        
                embedding = model(ids,mask)
                val_loss_batch = loss_func(embedding,intent_target)
        
        val_loss = val_loss_batch/float(num_batch)
        trial.report(val_loss, _)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss




study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000, timeout=300)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))