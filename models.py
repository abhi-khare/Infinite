import torch 
import torch.nn as nn 
from transformers import  DistilBertModel
from TorchCRF import CRF

class jointBert(nn.Module):

    def __init__(self, args):

        super(jointBert,self).__init__()
        
        # base encoder
        self.encoder = DistilBertModel.from_pretrained(args.model_weights,return_dict=True,output_hidden_states=True)

        # intent layer
        #p_intent = trial.suggest_float("intent_dropout", 0.1, 0.4)
        
        self.intent_dropout = nn.Dropout(args.intent_dropout_val)
        self.intent_linear = nn.Linear(768, args.intent_num)
        
        # slots layer
        self.slots_classifier = nn.Linear(768, args.slots_num)
        #p_slots = trial.suggest_float("slot_dropout", 0.1, 0.4)
        self.slots_dropout = nn.Dropout(args.slots_dropout_val)

        self.crf = CRF(args.slots_num)

        self.intent_loss = nn.CrossEntropyLoss()
        self.joint_loss_coef =  args.joint_loss_coef
    

    
    def forward(self, input_ids, attention_mask, intent_target, slots_target,slots_mask):

        encoded_output = self.encoder(input_ids, attention_mask)

        #intent data flow
        intent_hidden = encoded_output[0][:,0]
        #intent_LN = nn.LayerNorm(intent_hidden.size()[1:])
        intent_hidden = self.intent_dropout(intent_hidden)
        intent_logits = self.intent_linear(intent_hidden)
        # accumulating intent classification loss 
        intent_loss = self.intent_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)
        
        # slots data flow 
        slots_hidden = encoded_output[0]
        #slots_LN = nn.LayerNorm(slots_hidden.size()[1:])
        slots_logits = self.slots_classifier(self.slots_dropout(slots_hidden))#slots_LN(slots_hidden)))
        # accumulating slot prediction loss
        slots_loss = -1 * self.joint_loss_coef * self.crf(slots_logits, slots_target, mask=slots_mask.byte())
        slots_loss = torch.mean(slots_loss)
        
        joint_loss = (slots_loss + intent_loss)/2.0

        slots_pred = self.crf.viterbi_decode(slots_logits, slots_mask.byte())

        return joint_loss,slots_pred,intent_pred


class Bertencoder(nn.Module):

    def __init__(self,model):

        super(Bertencoder,self).__init__()
        
        self.encoder = DistilBertModel.from_pretrained(model,return_dict=True,output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(768, 768)
        
    
    def forward(self, input_ids, attention_mask):

        encoded_output = self.encoder(input_ids, attention_mask)
        hidden = self.pre_classifier(encoded_output[0][:,0])
        
        return hidden

