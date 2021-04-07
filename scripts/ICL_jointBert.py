import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import  DistilBertModel
from pytorch_metric_learning import losses

class ICL_IC_NER(nn.Module):

    def __init__(self, model_name):

        super(ICL_IC_NER,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model_name,return_dict=True,output_hidden_states=True)
        self.intent_dropout = nn.Dropout(0.3)
        self.intent_FC1 = nn.Linear(768, 512)
        self.intent_FC2 = nn.Linear(512, 128)
        self.intent_FC3 = nn.Linear(128, 18)
 

        # slots layer
        self.slots_dropout = nn.Dropout(0.3)
        self.slots_FC = nn.Linear(768, 159)
        

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss()
        self.log_vars = nn.Parameter(torch.zeros((3)))

        self.intent_CL = losses.NTXentLoss(temperature=0.07)
        

    
    def forward(self, input_ids, attention_mask , intent_target, slots_target,mode):

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
        loss_slots = torch.sum(precision2*slot_loss + self.log_vars[1],-1)

        ''' intent contrastive loss'''
        joint_loss = 0
        if mode=='TRAINING':
            ICLoss = self.intent_CL(intent_hidden,intent_target)

            precision3 = torch.exp(-self.log_vars[2])
            loss_slots = torch.sum(precision3*ICLoss + self.log_vars[2],-1)
           
            joint_loss = torch.mean(loss_intent + loss_slots + ICLoss)

            
        
        else:
            joint_loss = torch.mean(loss_intent + loss_slots)

        return {'joint_loss':joint_loss,
                'ic_loss': intent_loss,
                'ner_loss': slot_loss,
                'intent_pred':intent_pred,
                'slot_pred':slot_pred}

        
        #joint_loss = 0.5*intent_loss + 0.5*slot_loss