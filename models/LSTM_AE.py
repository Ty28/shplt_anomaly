# -*- coding:utf-8 -*-

"""
@File    : LSTM_AE.py
@Author  : ye tang
@IDE     : PyCharm
@Time    : 2022/01/23 16:40
@Function: 
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.device_count() >= 1 else "cpu")


class LSTM_Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTM_Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # multiple rnn encoder
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        # single rnn encoder
        # self.rnn = nn.LSTM(
        #     input_size=n_features,
        #     hidden_size=self.embedding_dim,
        #     num_layers=1,
        #     batch_first=True
        # )

    def forward(self, x):
        # multiple rnn encoder
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        # single rnn encoder
        # x, (hidden_n, _) = self.rnn(x)
        # x.shape = (batch_size * seq_len * n_features)
        # hidden_n.shape = (1 * batch_size * hidden_size)

        hidden_n = hidden_n.squeeze(dim=0)
        hidden_n = hidden_n.unsqueeze(1).repeat(1, self.seq_len, 1)
        return hidden_n


class LSTM_Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(LSTM_Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        # multiple rnn decoder
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

        # single rnn decoder
        # self.rnn = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=self.input_dim,
        #     num_layers=1,
        #     batch_first=True
        # )
        # self.output_layer_single = nn.Linear(self.input_dim, n_features)

    def forward(self, x):
        # multiple rnn decoder
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.output_layer(x)
        # single rnn decoder
        # x, (hidden_n, cell_n) = self.rnn(x)
        # return self.output_layer_single(x)


class LSTM_AutoEncoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.encoder = LSTM_Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = LSTM_Decoder(seq_len, embedding_dim, n_features).to(device)
        self._init_weight()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _init_weight(self):
        # Initiate parameters in the transformer model.
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
