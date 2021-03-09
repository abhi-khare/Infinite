import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformers import  DistilBertModel
from TorchCRF import CRF


class Bertencoder(nn.Module):

    def __init__(self,model):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 256)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden

class jointBert(nn.Module):

    def __init__(self, args):

        super(jointBert,self).__init__()
        
        self.model_mode = args.model_mode
        # base encoder
        self.encoder = DistilBertModel.from_pretrained(args.model_name,return_dict=True,output_hidden_states=True)
        
        if args.model_mode == 'IC_NER_MODE':
            # intent layer
            #p_intent = trial.suggest_float("intent_dropout", 0.1, 0.4)
            self.intent_dropout = nn.Dropout(args.intent_dropout_val)
            self.intent_linear_1 = nn.Linear(768, 64)
            self.intent_linear_2 = nn.Linear(64, args.intent_num)
            
            
            # slots layer
            self.slots_dropout = nn.Dropout(args.slots_dropout_val)
            self.slots_classifier_1 = nn.Linear(768, 256)
            self.slots_classifier_2 = nn.Linear(256, args.slots_num)
            #p_slots = trial.suggest_float("slots_dropout", 0.1, 0.4)
            

            self.crf = CRF(args.slots_num)

            self.intent_loss = nn.CrossEntropyLoss()
            
            self.log_vars = nn.Parameter(torch.zeros((2)))
            
            self.jlc = args.joint_loss_coef
        

    
    def forward(self, input_ids, attention_mask, intent_target, slots_target,slots_mask):

        encoded_output = self.encoder(input_ids, attention_mask)

        if self.model_mode == 'INTENT_CONTRA_MODE':
            classifier = torch.nn.Linear(768, 256)
            hidden = classifier(encoded_output[0][:,0])
            return hidden
        
        elif self.model_mode == 'SLOTS_CONTRA_MODE':
            return 2.0
        
        elif self.model_mode == 'INTENT_SLOTS_CONTRA_MODE':

            return 3.0
        
        elif self.model_mode == 'IC_NER_MODE':

            #intent data flow
            intent_hidden = encoded_output[0][:,0]
            intent_hidden = self.intent_linear_1(self.intent_dropout(F.relu(intent_hidden)))
            intent_logits = self.intent_linear_2(F.relu(intent_hidden))
            # accumulating intent classification loss 
            intent_loss = self.intent_loss(intent_logits, intent_target)
            intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)
            
            # slots data flow 
            slots_hidden = encoded_output[0]
            slots_hidden = self.slots_classifier_1(self.slots_dropout(F.relu(slots_hidden)))
            slots_logits = self.slots_classifier_2(F.relu(slots_hidden))

            # accumulating slot prediction loss
            slots_loss = -1 * self.crf(slots_logits, slots_target, mask=slots_mask.byte())
            slots_loss = torch.mean(slots_loss)
              
            joint_loss = ((1-self.jlc)*intent_loss + (self.jlc)*slots_loss)

            slots_pred = self.crf.viterbi_decode(slots_logits, slots_mask.byte())

            return joint_loss,slots_pred,intent_pred,intent_loss,slots_loss


