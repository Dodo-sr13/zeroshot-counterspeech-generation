from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel

import torch.nn as nn
import torch



class BertForRegression(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.loss_fct = nn.MSELoss()
        self.init_weights()
        
#     ###uncomment to freeze bert the parameters
#         for param in self.bert.parameters():
#             param.requires_grad = False
              


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        

        logits = self.classifier(pooled_output)

        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    
    

class RobertaForRegression(RobertaPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.MSELoss()
        self.init_weights()
        
#         #         uncomment to freeze bert the parameters
#         for param in self.roberta.parameters():
#             param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
 
class HateAlert(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])

        self.classifier = nn.Linear(config.hidden_size, 2)
        
        self.classifier1 = nn.Linear(2, 1)
        
        self.relu = nn.ReLU()
        self.loss_fct = nn.MSELoss()
        self.init_weights()
        
# #         uncomment to freeze bert the parameters
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         self.classifier.requires_grad = False   
              


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        

        logits = self.classifier(pooled_output)
        #
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.classifier1(logits)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

