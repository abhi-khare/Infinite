import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from transformers import DistilBertModel


class jointBert(nn.Module):
    def __init__(self, args):

        super(jointBert, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
            args.encoder, return_dict=True, output_hidden_states=True,
            sinusoidal_pos_embds=True
        )
        
        self.intent_head = nn.Sequential(nn.GELU(),
                                         nn.Dropout(args.intent_dropout),
                                         nn.Linear(768,args.intent_hidden),
                                         nn.GELU(),
                                         nn.Linear(args.intent_hidden,args.intent_count) 
                                        )

        self.slots_head = nn.Sequential(nn.GELU(),
                                         nn.Dropout(args.slots_dropout),
                                         nn.Linear(768,args.slots_hidden),
                                         nn.GELU(),
                                         nn.Linear(args.slots_hidden,args.slots_count) 
                                        )

        self.CE_loss = nn.CrossEntropyLoss()
        self.jointCoef = args.jointCoef
        self.args = args

    def forward(self, input_ids, attention_mask, intent_target, slots_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_logits = self.intent_head(intent_hidden)

        # slots data flow
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_head(slots_hidden)

        # accumulating intent classification loss
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        # accumulating slot prediction loss
        slot_loss = self.CE_loss(
            slots_logits.view(-1, self.args.slots_count), slots_target.view(-1)
        )
        slot_pred = torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)

        joint_loss = self.jointCoef * intent_loss + (1.0 - self.jointCoef) * slot_loss

        return {
            "joint_loss": joint_loss,
            "ic_loss": intent_loss,
            "ner_loss": slot_loss,
            "intent_pred": intent_pred,
            "slot_pred": slot_pred,
        }
