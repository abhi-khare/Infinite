import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import  DistilBertModel


class IC_NER(nn.Module):

    def __init__(self, model_name):

        super(IC_NER,self).__init__()
        
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
        self.log_vars = nn.Parameter(torch.zeros((2)))
        

    
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
        slot_pred =  torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        # accumulating slot prediction loss
        slot_loss = self.slot_loss_fn(slots_logits.view(-1, 159), slots_target.view(-1))


        '''Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics'''
        
        precision1 = torch.exp(-self.log_vars[0])
        loss_intent = torch.sum(precision1*intent_loss + self.log_vars[0],-1)

        precision2 = torch.exp(-self.log_vars[1])
        loss_slots = torch.sum(precision1*slot_loss + self.log_vars[1],-1)

        joint_loss = torch.mean(loss_intent + loss_slots)
        

        return {'joint_loss':joint_loss,
                'ic_loss': intent_loss,
                'ner_loss': slot_loss,
                'intent_pred':intent_pred,
                'slot_pred':slot_pred}
