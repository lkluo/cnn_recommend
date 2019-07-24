"""CNN text classification"""
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class CNNText(nn.Module):
    
    def __init__(self,
                 embed_size,
                 num_class,
                 filter_size,
                 kernel_size,
                 dropout=0.1):
        super(CNNText, self).__init__()

        Ci = 1
        Co = filter_size
        Ks = kernel_size
        D = embed_size
        C = num_class

        if isinstance(Ks, int):
            Ks = [Ks]

        """2d CNN"""
        # input: [batch_size, seq_len, embed_size]
        # output:
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        # self.convs = nn.Sequential(
        #     nn.Conv2d(1, kernel_num, (kernerl_size, embed_size)),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernerl_size)
        # )

        """dropout layer"""
        self.dropout = nn.Dropout(dropout)
        """fully-connected layer"""
        self.ffc = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        # use pre-trained word2vec
        x = Variable(x)

        # make it to [batch_size, 1, seq_len, embed_size]
        x = x.unsqueeze(1)

        # [(batch_size, output_size, seq_len)] * len(Ks)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # [(batch_size, output_size)] * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # [batch_size, len(Ks)*Co)]
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # [batch_size, num_class]
        logits = self.ffc(x)
        # [num_class, 1]
        # probs = F.softmax(logits, 1)
        # predict = torch.max(probs, 1)[1]

        return logits