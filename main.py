from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from input_variational_dropout import InputVariationalDropout


def count_parameters(model):
    #  torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
    parameters_count = ["Name : {}\t\tCount: {}".format(name, para.numel()) for name, para in model.named_parameters()
                        if para.requires_grad]

    return "\n".join(parameters_count)


class myModel(nn.Module):
    def __int__(self,
                tokenizer: PreTrainedTokenizerBase,
                lstm_input_size: int,
                lstm_hidden_size: int,
                lstm_bidirectional: bool,
                lstm_layers: int,
                pretrained_model: str,
                inp_drop_rate: float = 0.2,
                out_drop_rate: float = 0.2,
                super_mode: str = "before",
                backbone: str = "unet",
                unet_down_channel: int = 256,
                feature_sel: int = 127
                ):
        super(myModel, self).__init__()

        self.text_encoder = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=lstm_bidirectional
        )

        self.var_inp_dropout = (lambda x: x,
                                InputVariationalDropout(p=inp_drop_rate))[inp_drop_rate > 0.0]

        self.var_out_dropout = (lambda x: x,
                                InputVariationalDropout(p=inp_drop_rate))[out_drop_rate > 0.0]

        self.hidden_size = self.text_encoder.get_ou