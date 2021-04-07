import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import  DistilBertModel


class IC_NER(nn.Module):

    def __init__(self, args):

        super(IC_NER,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(args.model_name,return_dict=True,output_hidden_states=True)
       
        self.intent_dropout_1 = nn.Dropout(0.30)
        self.intent_dropout_2 = nn.Dropout(0.15)
        self.intent_FC1 = nn.Linear(768, 128)
        self.intent_FC2 = nn.Linear(128, args.intent_num)
 

        # slots layer
        self.slots_dropout = nn.Dropout(0.30)
        self.slots_FC = nn.Linear(768, args.slots_num)
        
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss()

        self.jlc = args.joint_loss_coef
        self.args = args
        

    
    def forward(self, input_ids, attention_mask , intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        #intent data flow
        intent_hidden = encoded_output[0][:,0]
        intent_hidden = self.intent_FC1(self.intent_dropout_1(F.gelu(intent_hidden)))
        intent_logits = self.intent_FC2(self.intent_dropout_2(F.gelu(intent_hidden)))
        
        
        # accumulating intent classification loss 
        intent_loss = self.intent_loss_fn(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)
        

        # slots data flow 
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_FC(self.slots_dropout(F.relu(slots_hidden)))
        slot_pred =  torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        # accumulating slot prediction loss
        slot_loss = self.slot_loss_fn(slots_logits.view(-1, self.args.slots_num), slots_target.view(-1))


        '''Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics'''
        
        #precision1 = torch.exp(-self.log_vars[0])
        #loss_intent = torch.sum(precision1*intent_loss + self.log_vars[0],-1)

        #precision2 = torch.exp(-self.log_vars[1])
        #loss_slots = torch.sum(precision1*slot_loss + self.log_vars[1],-1)

        #joint_loss = torch.mean(loss_intent + loss_slots)
        
        joint_loss = self.jlc*intent_loss + (1.0 - self.jlc)*slot_loss

        return {'joint_loss':joint_loss,
                'ic_loss': intent_loss,
                'ner_loss': slot_loss,
                'intent_pred':intent_pred,
                'slot_pred':slot_pred}
