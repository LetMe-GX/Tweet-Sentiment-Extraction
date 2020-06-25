import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import ElectraConfig
from transformers.modeling_electra import ElectraPreTrainedModel, ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP, ElectraModel
from seq2seq import seq2seq_rnn as sq
from seq2seq import highway as hw
import numpy as np

def dist_between(start_logits, end_logits, max_seq_len=192):
    """get dist btw. pred & ground_truth"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False)
    linear_func = linear_func.to(device)

    start_pos = (start_logits * linear_func).sum(axis=1)
    end_pos = (end_logits * linear_func).sum(axis=1)

    diff = end_pos - start_pos

    return diff.sum(axis=0) / diff.size(0)


def dist_loss(start_logits, end_logits, start_positions, end_positions, max_seq_len=192, scale=1):
    """calculate distance loss between prediction's length & GT's length

    Input
    - start_logits ; shape (batch, max_seq_len{128})
        - logits for start index
    - end_logits
        - logits for end index
    - start_positions ; shape (batch, 1)
        - start index for GT
    - end_positions
        - end index for GT
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_logits = torch.nn.Softmax(1)(start_logits)  # shape ; (batch, max_seq_len)
    end_logits = torch.nn.Softmax(1)(end_logits)

    start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_seq_len).to(device)
    end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_seq_len).to(device)

    pred_dist = dist_between(start_logits, end_logits, max_seq_len)
    gt_dist = dist_between(start_one_hot, end_one_hot, max_seq_len)  # always positive
    diff = (gt_dist - pred_dist)

    rev_diff_squared = 1 - torch.sqrt(diff * diff)  # as diff is smaller, make it get closer to the one
    loss = -torch.log(
        rev_diff_squared)  # by using negative log function, if argument is near zero -> inifinite, near one -> zero

    return loss * scale


def cal_loss(pred, target, ignore_index=None, smoothing=0.):
    if smoothing > 0:
        log_prob = pred.log_softmax(dim=-1)
        with torch.no_grad():
            weight = pred.new_ones(pred.size()) * smoothing / (pred.size(-1) - 1.)
            weight.scatter_(-1, target.unsqueeze(-1), (1. - smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
    else:
        loss_fuc = CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fuc(pred, target)
    return loss


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    """
    Identical to BertForQuestionAnswering other than using an ElectraModel
    """

    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "electra"

    def __init__(self, config, weight=None):
        config.output_hidden_states = True
        super().__init__(config)
        self.num_labels = config.num_labels
        self.smoothing = 0
        self.dist_loss = False
        self.electra = ElectraModel(config)

        self.encoder = sq.Encoder('GRU',config.hidden_size,config.hidden_size,0.3)
        self.decoder = sq.Decoder('GRU',config.hidden_size,config.hidden_size,config.hidden_size,0.3)

        self.seq2seq = sq.Seq2seq(self.encoder, self.decoder)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.drop_out = nn.Dropout(0.3)

        self.highway = hw.Highway(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        # outputs = [batch, seq_len, emb]
        seq_scr = outputs[0].permute(1,0,2)
        # seq_sct = [seq_len, batch, emb]
        seq_output = self.seq2seq(seq_scr, seq_scr)
        # seq_output [seq_len, batch, emb]
        seq_output = seq_output.permute(1,0,2)
        # seq_output [batch seq_len emb]
        hw_out = self.highway(outputs[0],seq_output)
        # seq_output [batch seq_len emb]
        sequence_output = self.drop_out(hw_out)
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            start_loss = cal_loss(start_logits, start_positions, ignore_index=ignored_index, smoothing=self.smoothing)
            end_loss = cal_loss(end_logits, end_positions, ignore_index=ignored_index, smoothing=self.smoothing)
            if self.dist_loss:
                d_loss = dist_loss(start_logits, end_logits, start_positions, end_positions)
                total_loss = (start_loss + end_loss) / 2 + d_loss * 2
            else:
                total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
