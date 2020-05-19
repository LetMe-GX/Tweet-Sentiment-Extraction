# https://github.com/bentrevett/
# LISENCE: MIT

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=hid_dim,
                          bidirectional=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(src)
        # embedded = [seq_len, batch, emb]
        outputs, hidden = self.rnn(embedded)
        # outputs = [seq_len, batch, num_directions*hidden_size]
        # hidden = [num_layers*num_directions,batch,hidden_size]
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size=(emb_dim+2*hid_dim),
                          hidden_size=hid_dim,
                          bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

        self.fc_out = nn.Linear(emb_dim+hid_dim*2+hid_dim*2, output_dim)

    def forward(self, input_str, hidden, context):
        # input_str = [batch size, emb_dim]
        # hidden = [2, batch size, hid dim]
        # context = [2, batch size, hid dim]

        embedded = self.dropout(input_str)
        # embedded = [batch, emb]
        emb_con = torch.cat((embedded, hidden[0], hidden[1]), dim=1)
        # emb_con = [batch size, emb_dim + 2*hid_dim]
        emb_con = emb_con.unsqueeze(0)
        # emb_con = [1, batch, emb+2*hid]

        output, hidden = self.rnn(emb_con, hidden)
        # output = [1, batch, num_directions*hid_dim]
        # hidden = [2, batch, hid_dim]
        output = torch.cat((embedded,
                            hidden[0],
                            hidden[1],
                            context[0],
                            context[1]),
                            dim=1)
        # output = [batch, emb+hid+hid+hid+hid]
        prediction = self.fc_out(output)
        # prediction = [batch, output_dim]
        return prediction, hidden


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src = [seq_len, batch, emb]
        # trg = [seq_len, batch, emb]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # trg_vocab_size = config.hidden_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(torch.device('cuda'))
        context = self.encoder(src)
        # context = [2, batch, hid_dim]
        hidden = context
        # hidden = [2, batch hid_dim]

        for t in range(trg_len):
            input = trg[t]
            # input = [batch, hid_dim]
            output, hidden = self.decoder(input, hidden, context)
            # output = [batch, config.hidden_size]
            outputs[t] = output
            input = trg[t]

        return outputs
