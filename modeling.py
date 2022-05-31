import math
import torch
import logging
from torch import nn
import numpy as np
from transformers.models.bert.modeling_bert import (BertModel, BertPreTrainedModel)
logger = logging.getLogger(__name__)


class RankingBERT_Train(BertPreTrainedModel):
    def __init__(self, config):
        super(RankingBERT_Train, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids,
                position_ids, labels=None):

        attention_mask = (input_ids != 0)

        bert_pooler_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)[1]

        output = self.out(self.dropout(bert_pooler_output))
        # shape = [B, 1]

        if labels is not None:

            loss_fct = nn.MarginRankingLoss(margin=1.0, reduction='mean')

            y_pos, y_neg = [], []
            for batch_index in range(len(labels)):
                label = labels[batch_index]
                if label > 0:
                    y_pos.append(output[batch_index])
                else:
                    y_neg.append(output[batch_index])
            y_pos = torch.cat(y_pos, dim=-1)
            y_neg = torch.cat(y_neg, dim=-1)
            y_true = torch.ones_like(y_pos)
            assert len(y_pos) == len(y_neg)

            loss = loss_fct(y_pos, y_neg, y_true)
            output = loss, *output
        return output
