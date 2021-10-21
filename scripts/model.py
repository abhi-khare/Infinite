import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

class intent_classifier(nn.Module):
    def __init__(self, args):

        super(intent_classifier, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
                                        args.encoder, 
                                        return_dict=True, 
                                        output_hidden_states=True,
                                        sinusoidal_pos_embds=True, 
                                        cache_dir='/efs-storage/research/model/'
                                    )
        
        self.intent_head = nn.Sequential(
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(args.intent_dropout),
                                        nn.Linear(768, args.intent_hidden),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(args.intent_dropout),
                                        nn.Linear(args.intent_hidden, args.num_class)
                                        )

        self.CE_loss = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, input_ids, attention_mask, intent_target):

        encoded_output = self.encoder(input_ids, attention_mask)

        # intent data flow
        intent_hidden = encoded_output[0][:, 0]
        intent_logits = self.intent_head(intent_hidden)

        # accumulating intent classification loss
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        return {
            "ic_loss": intent_loss,
            "intent_pred": intent_pred
        }