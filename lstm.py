import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTM(nn.Module):
    def __int__(self,
                input_size: int,
                hidden_size: int,
                num_layers: int = 1,
                bias: bool = True,
                dropout: float = 0.0,
                bidirectional: bool = False,
                ):

        self._module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        try:
            if not self._module.batch_first:
                raise ValueError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    def forward(self, feature, seq_len) -> torch.Tensor:
        """
        因为使用的是 rnn-like 模型, 这里需要 pack一下数据, 才能输入 rnn-like 模型.
        在 pack 数据之前, 需要两个步骤: 1. 依照长度降序排序  2. 计算batch内每个sample的长度
        两个步骤 这里在我们在 DataLoader 类里进行实现.
                实现方式为: examples: List[Tuple(List[int], int)]
                    def collate_fn(examples):
                        examples.sort(key=lambda x: len(x[0]), reverse=True)
                        seq_len = torch.tensor([len(example[0]) for example in examples])
                        feature = [torch.tensor(example[0]) for example in examples]
                        labels = torch.tensor([example[1] for example in examples])
                        feature = pad_sequence(feature, batch_first=True, padding_value=0)
                        return feature, labels, seq_len
        当前函数 主要分 3个部分: 1. pack data  2. 输入模型  3. unpack data
                实现方式为:
                    1. pack = pack_padded_sequence(feature, seq_len, batch_first=True)
                    2. output, _ = rnn-like(pack, None)
                    3. unpack, _ = pad_packed_sequence(output, batch_first=True)

        参考: https://colab.research.google.com/drive/1cpn6pk2J4liha9jgDLNWhEWeWJb2cdch?usp=sharing#scrollTo=soc-PApFBmyC
             https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        """

        pack = pack_padded_sequence(feature, seq_len, batch_first=True)

        output, _ = self._module(pack, None)

        unpack, _ = pad_packed_sequence(output, batch_first=True)

        """
        out_forward = pack[range(len(pack)), seq_len - 1, :self.get_input_dim()]
        out_reverse = pack[:, 0, self.get_input_dim():]
        """
        return unpack
