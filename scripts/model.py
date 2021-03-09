import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import  DistilBertModel
from TorchCRF import CRF


class jointBert(nn.Module):

    def __init__(self, model_name):

        super(jointBert,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model_name,return_dict=True,output_hidden_states=True)
        #p_intent = trial.suggest_float("intent_dropout", 0.1, 0.4)
        self.intent_dropout = nn.Dropout(0.1)#args.intent_dropout_val)
        self.intent_FC = nn.Linear(768, 17)
 

        # slots layer
        self.slots_dropout = nn.Dropout(0.1)#args.slots_dropout_val)
        self.slots_FC = nn.Linear(768, 159)
        #p_slots = trial.suggest_float("slots_dropout", 0.1, 0.4)

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss()
        #self.log_vars = nn.Parameter(torch.zeros((2)))

        self.jlc = 0.5#args.joint_loss_coef
        

    
    def forward(self, input_ids, attention_mask , intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        #intent data flow
        intent_hidden = encoded_output[0][:,0]
        intent_logits = self.intent_FC(self.intent_dropout(F.relu(intent_hidden)))
        
        
        # accumulating intent classification loss 
        intent_loss = self.intent_loss_fn(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)
        

        # slots data flow 
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_FC(self.slots_dropout(F.relu(slots_hidden)))
       

        # accumulating slot prediction loss
        slot_loss = self.slot_loss_fn(slots_logits.view(-1, 159), slots_target.view(-1))
        
        joint_loss = ((1-self.jlc)*intent_loss + (self.jlc)*slot_loss)
        

        return joint_loss,intent_pred,intent_loss,slot_loss
