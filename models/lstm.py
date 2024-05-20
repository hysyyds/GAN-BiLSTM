#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


# Att-BiLSTM model
class Attention(nn.Module):
    def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


class AttBiLSTM(nn.Module):
    def __init__(
            self,
            n_classes: int,
            emb_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.attention = Attention(rnn_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
        self.batchnorm = nn.BatchNorm1d(emb_size)
        self.layernorm = nn.LayerNorm(rnn_size)

    def forward(self, x):
        size = x.shape[0]
        # 创建副对角线元素
        sub_diag_0 = torch.ones(size)
        sub_diag_1 = torch.ones(size - 1)
        sub_diag_2 = torch.ones(size - 2)
        sub_diag_3 = torch.ones(size - 1)
        sub_diag_4 = torch.ones(size - 2)
        # 创建对应的副对角线张量
        tensor0 = torch.diag(sub_diag_0, diagonal=0)
        tensor1 = torch.diag(sub_diag_1, diagonal=1)
        tensor2 = torch.diag(sub_diag_2, diagonal=2)
        tensor3 = torch.diag(sub_diag_3, diagonal=-1)
        tensor4 = torch.diag(sub_diag_4, diagonal=-2)
        # 将两条副对角线张量相加
        adj = tensor0 + tensor1 + tensor2 + tensor3 +tensor4
        adj=adj.cuda()
        denom = adj.sum(-1, keepdim=True)
        # (batch_size, time,emb_size)
        x.transpose_(1, 2)
        # (batch_size, emb_size,time)
        x=self.batchnorm(x)
        # (batch_size, time,emb_size)
        x.transpose_(1, 2)
        rnn_out, _ = self.BiLSTM(x)
        H = rnn_out[:, :, : self.rnn_size] + (adj @ rnn_out[:, :, self.rnn_size:].squeeze()).unsqueeze(1)
        H = self.layernorm(H)
        # attention module
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)
        h = self.tanh(r) /denom # (batch_size, rnn_size)
        scores = self.classifier(h)
        return scores
