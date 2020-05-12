import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import ElectraConfig
from transformers.modeling_electra import ElectraPreTrainedModel, ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP, ElectraModel


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
        self.combine_hidden = config.combine_hidden
        self.electra = ElectraModel(config)
        self.drop_out = nn.Dropout(config.drop_out)
        self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)

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
        sequence_output = outputs[0]
        all_hidden_states = outputs[1]
        if self.combine_hidden:
            sequence_output = torch.cat((all_hidden_states[-1], all_hidden_states[-2]), dim=-1)
        else:
            sequence_output = self.drop_out(sequence_output)
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

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)