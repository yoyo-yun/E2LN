import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_zoo import LN_based_attention, SelfAttention
from models.classifier import Classifier

from models.bert_for_doc_all import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
class BertForSequenceClassification_doc(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.cus_config = kwargs['cus_config']
        self.num_labels = config.num_labels
        self.config = config
        self.pooling = self.cus_config.pooling

        self.bert = BertModel(config, self.cus_config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        assert self.pooling in ["cls", "avg", "max", "la", "lnla"], "wrong pooling"
        if self.pooling == "la":
            self.attention = SelfAttention(config.hidden_size, self.cus_config)
            self.cus_config.num_classes = self.num_labels
            self.classifier = Classifier(config.hidden_size, self.cus_config)
        if self.pooling == 'lnla':
            self.set_lnla()
            self.cus_config.num_classes = self.num_labels
            self.classifier = Classifier(config.hidden_size, self.cus_config)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        user_ids=None,
        item_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            user_ids=user_ids,
            item_ids=item_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.pooling == 'cls':
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
        elif self.pooling == 'max':
            pooled_output = outputs[0] # (bs, seq, dim)
            pooled_output, _ = pooled_output.max(1)
        elif self.pooling == "avg":
            pooled_output = outputs[0]  # (bs, seq, dim)
            pooled_output = pooled_output.mean(1)
        elif self.pooling == "la":
            pooled_output = outputs[0]  # (bs, seq, dim)
            pooled_output = self.attention(pooled_output)
        else:
            pooled_output = outputs[0]  # (bs, seq, dim)

        logits = self.classifier(pooled_output)

        loss = None
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_lnla(self):
        self.bert.encoder.layer[-1].output.LayerNorm.pooling = True
