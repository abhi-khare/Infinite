import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel


class jointBert(nn.Module):
    def __init__(self, args):

        super(jointBert, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
            args.encoder, return_dict=True, output_hidden_states=True,
            sinusoidal_pos_embds=True, cache_dir='/efs-storage/model/'
        )
        
        self.intent_head = nn.Sequential(
                                         nn.Dropout(args.intent_dropout),
                                         nn.Linear(768,args.intent_count)
                                        )

        self.slots_head = nn.Sequential(
                                         nn.Dropout(args.slots_dropout),
                                         nn.Linear(768,args.slots_count)
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

class hierCon_model(nn.Module):

    def __init__(self, args):

        super(hierCon_model, self).__init__()

        self.encoder = DistilBertModel.from_pretrained(
            args.encoder, return_dict=True, output_hidden_states=True,
            sinusoidal_pos_embds=True, cache_dir='/efs-storage/model/'
        )
        
        self.intent_head = nn.Sequential(
                                         nn.Dropout(args.intent_dropout),
                                         nn.Linear(768,args.intent_count)
                                        )

        self.slots_head = nn.Sequential(
                                         nn.Dropout(args.slots_dropout),
                                         nn.Linear(768,args.slots_count)
                                        )

        self.token_contrast_proj = nn.Sequential(
                                                 nn.Linear(768,768),
                                                 nn.BatchNorm1d(768),
                                                 nn.GELU(),
                                                 nn.Linear(768,768) 
                                                )
        
        self.sent_contrast_proj = nn.Sequential(
                                                 nn.Linear(768,768),
                                                 nn.BatchNorm1d(768),
                                                 nn.GELU(),
                                                 nn.Linear(768,768) 
                                                )
        
        self.criterion = nn.CosineSimilarity(dim=1)
        
        self.CE_loss = nn.CrossEntropyLoss()

        self.icnerCoef = args.icnerCoef
        self.hierConCoef = args.hierConCoef
        self.args = args

    def ICNER_loss(self, encoded_output, intent_target, slots_target):

        # intent prediction loss
        intent_hidden = encoded_output[0][:, 0]
        intent_logits = self.intent_head(intent_hidden)
        intent_loss = self.CE_loss(intent_logits, intent_target)
        intent_pred = torch.argmax(nn.Softmax(dim=1)(intent_logits), axis=1)

        # slots prediction loss
        slots_hidden = encoded_output[0]
        slots_logits = self.slots_head(slots_hidden)
        slots_pred = torch.argmax(nn.Softmax(dim=2)(slots_logits), axis=2)
        slots_loss = self.CE_loss(
            slots_logits.view(-1, self.args.slots_count), slots_target.view(-1)
        )

        joint_loss = self.icnerCoef * intent_loss + (1.0 - self.icnerCoef) * slots_loss

        return {
            "joint_loss": joint_loss,
            "ic_loss": intent_loss,
            "ner_loss": slots_loss,
            "intent_pred": intent_pred,
            "slot_pred": slots_pred,
        }

    def sentCL(self, sentz1, sentz2):

        # calculating sentence level loss
        p1, p2 = self.sent_contrast_proj(sentz1), self.sent_contrast_proj(sentz2)
        sentz2.detach()
        sentz1.detach()
        
        sentCLLoss =  -(self.criterion(p2, sentz1).mean() + self.criterion(p1, sentz2).mean()) * 0.5

        return sentCLLoss
    
    def tokenCL(self, tokenEmb1,tokenEmb2,tokenID1,tokenID2):
        #torch.Size([32, 56, 768]) torch.Size([32, 56, 768]) torch.Size([1792]) torch.Size([1792])
        tokenID1 = torch.flatten(tokenID1)
        tokenID2 = torch.flatten(tokenID2)
        
        shape = tokenEmb1.shape
        tokenEmb1 = tokenEmb1.view(shape[0]*shape[1],-1)
        tokenEmb2 = tokenEmb2.view(shape[0]*shape[1],-1) #torch.Size([1792, 768]) torch.Size([1792, 768])
        
        filterTokenIdx1 = [idx for idx,val in enumerate(tokenID1.tolist()) if (val==-100 or val == 2000)!=True]
        filterTokenIdx2 = [idx for idx,val in enumerate(tokenID2.tolist()) if (val==-100 or val == 2000)!=True]
        print(len(filterTokenIdx1),len(filterTokenIdx2))
        if len(filterTokenIdx1) > 0:
            filterTokenIdx1 = torch.tensor( filterTokenIdx1,dtype=torch.long,device=torch.device('cuda'))
            tokenEmb1 = torch.index_select(tokenEmb1,0,filterTokenIdx1) 
        
        if len(filterTokenIdx2) > 0:
            filterTokenIdx2 = torch.tensor( filterTokenIdx2,dtype=torch.long,device=torch.device('cuda')) 
            tokenEmb2 = torch.index_select(tokenEmb2,0,filterTokenIdx2)
        
        # calculating sentence level loss
        p1, p2 = self.token_contrast_proj(tokenEmb1), self.token_contrast_proj(tokenEmb2)
        tokenEmb1.detach()
        tokenEmb2.detach()
        tokenCLLoss =  -(self.criterion(p2, tokenEmb1).mean() + self.criterion(p1, tokenEmb2).mean()) * 0.5
            
        return tokenCLLoss

    def forward(self, batch , mode):

        if mode == "ICNER":
            encoded_output = self.encoder(batch['supBatch']['token_ids'], batch['supBatch']['mask'])
            return self.ICNER_loss(encoded_output, batch['supBatch']['intent_id'], batch['supBatch']['slots_id'])

        if mode == "hierCon":
            encoded_output_0 = self.encoder(batch['HCLBatch'][0]['token_ids'],batch['HCLBatch'][0]['mask']) 
            encoded_output_1 = self.encoder(batch['HCLBatch'][1]['token_ids'],batch['HCLBatch'][1]['mask']) 
            sentCL = self.sentCL(encoded_output_0[0][:, 0], encoded_output_1[0][:, 0])
            
            tokenIDs0 = batch['HCLBatch'][0]['token_id']
            tokenIDs1 = batch['HCLBatch'][1]['token_id']
            
            tokenCL = self.tokenCL(encoded_output_0[0], encoded_output_1[0],tokenIDs0,tokenIDs1)

            hierConLoss = self.args.hierConCoef*sentCL + (1.0-self.args.hierConCoef)*tokenCL
            
            return hierConLoss
