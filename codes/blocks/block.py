#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCKS_LIB = {
    'MLP-16':
        lambda params: MLP('MLP-16', 16, params),
    'MLP-32':
        lambda params: MLP('MLP-32', 32, params),
    'MLP-64':
        lambda params: MLP('MLP-64', 64, params),
    'MLP-128':
        lambda params: MLP('MLP-128', 128, params),
    'MLP-256':
        lambda params: MLP('MLP-256', 256, params),
    'MLP-512':
        lambda params: MLP('MLP-512', 512, params),
    'MLP-1024':
        lambda params: MLP('MLP-1024', 1024, params),
    'ElementWise-sum':
        lambda params: ElementWise('ElementWise-sum', 'sum', params),
    'ElementWise-avg':
        lambda params: ElementWise('ElementWise-avg', 'avg', params),
    'ElementWise-min':
        lambda params: ElementWise('ElementWise-min', 'min', params),
    'ElementWise-max':
        lambda params: ElementWise('ElementWise-max', 'max', params),
    'ElementWise-innerproduct':
        lambda params: ElementWise('ElementWise-innerproduct', 'innerproduct', params),
    'SelfAttention-1':
        lambda params: SelfAttention('SelfAttention-1', 1, params),
    'SelfAttention-2':
        lambda params: SelfAttention('SelfAttention-2', 2, params),
    'SelfAttention-3':
        lambda params: SelfAttention('SelfAttention-3', 3, params),
    'SelfAttention-4':
        lambda params: SelfAttention('SelfAttention-4', 4, params),
}

BLOCK2ID = {blk: i for i, blk in enumerate(sorted(BLOCKS_LIB.keys()))}
ID2BLOCK = {i: blk for i, blk in enumerate(sorted(BLOCKS_LIB.keys()))}



BLOCK_FLOPS = {'0_MLP-16': (1056.0, 2256.0), '0_MLP-32': (2080.0, 4320.0), '0_MLP-64': (4128.0, 8448.0),
               '0_MLP-128': (8224.0, 16704.0), '0_MLP-256': (16416.0, 33216.0),
               '0_MLP-512': (32800.0, 66240.0), '0_MLP-1024': (65568.0, 132288.0),
               '0_SelfAttention-1': (1320.0, 2784.0), '0_SelfAttention-2': (2608.0, 5376.0),
               '0_SelfAttention-3': (3896.0, 7968.0), '0_SelfAttention-4': (5184.0, 10560.0),
               '1_MLP-16': (1568.0, 3280.0), '1_MLP-32': (3104.0, 6368.0), '1_MLP-64': (6176.0, 12544.0),
               '1_MLP-128': (12320.0, 24896.0), '1_MLP-256': (24608.0, 49600.0),
               '1_MLP-512': (49184.0, 99008.0), '1_MLP-1024': (98336.0, 197824.0),
               '1_ElementWise-sum': (2080.0, 4288.0), '1_ElementWise-avg': (2112.0, 4288.0),
               '1_ElementWise-min': (2080.0, 4288.0), '1_ElementWise-max': (2080.0, 4288.0),
               '1_ElementWise-innerproduct': (2144.0, 4288.0), '1_SelfAttention-1': (2624.0, 3296.0),
               '1_SelfAttention-2': (5216.0, 6400.0), '1_SelfAttention-3': (7808.0, 9504.0),
               '1_SelfAttention-4': (10400.0, 12608.0), '2_MLP-16': (2080.0, 4304.0),
               '2_MLP-32': (4128.0, 8416.0), '2_MLP-64': (8224.0, 16640.0), '2_MLP-128': (16416.0, 33088.0),
               '2_MLP-256': (32800.0, 65984.0), '2_MLP-512': (65568.0, 131776.0),
               '2_MLP-1024': (131104.0, 263360.0), '2_ElementWise-sum': (2080.0, 4288.0),
               '2_ElementWise-avg': (2112.0, 4288.0), '2_ElementWise-min': (2080.0, 4288.0),
               '2_ElementWise-max': (2080.0, 4288.0), '2_ElementWise-innerproduct': (2176.0, 4288.0),
               '2_SelfAttention-1': (3944.0, 3808.0), '2_SelfAttention-2': (7856.0, 7424.0),
               '2_SelfAttention-3': (11768.0, 11040.0), '2_SelfAttention-4': (15680.0, 14656.0),
               '3_MLP-16': (2592.0, 5328.0), '3_MLP-32': (5152.0, 10464.0), '3_MLP-64': (10272.0, 20736.0),
               '3_MLP-128': (20512.0, 41280.0), '3_MLP-256': (40992.0, 82368.0),
               '3_MLP-512': (81952.0, 164544.0), '3_MLP-1024': (163872.0, 328896.0),
               '3_ElementWise-sum': (2080.0, 4288.0), '3_ElementWise-avg': (2112.0, 4288.0),
               '3_ElementWise-min': (2080.0, 4288.0), '3_ElementWise-max': (2080.0, 4288.0),
               '3_ElementWise-innerproduct': (2208.0, 4288.0), '3_SelfAttention-1': (5280.0, 4320.0),
               '3_SelfAttention-2': (10528.0, 8448.0), '3_SelfAttention-3': (15776.0, 12576.0),
               '3_SelfAttention-4': (21024.0, 16704.0), '4_MLP-16': (3104.0, 6352.0),
               '4_MLP-32': (6176.0, 12512.0), '4_MLP-64': (12320.0, 24832.0), '4_MLP-128': (24608.0, 49472.0),
               '4_MLP-256': (49184.0, 98752.0), '4_MLP-512': (98336.0, 197312.0),
               '4_MLP-1024': (196640.0, 394432.0), '4_ElementWise-sum': (2080.0, 4288.0),
               '4_ElementWise-avg': (2112.0, 4288.0), '4_ElementWise-min': (2080.0, 4288.0),
               '4_ElementWise-max': (2080.0, 4288.0), '4_ElementWise-innerproduct': (2240.0, 4288.0),
               '4_SelfAttention-1': (6632.0, 4832.0), '4_SelfAttention-2': (13232.0, 9472.0),
               '4_SelfAttention-3': (19832.0, 14112.0), '4_SelfAttention-4': (26432.0, 18752.0),
               '5_MLP-16': (3616.0, 7376.0), '5_MLP-32': (7200.0, 14560.0), '5_MLP-64': (14368.0, 28928.0),
               '5_MLP-128': (28704.0, 57664.0), '5_MLP-256': (57376.0, 115136.0),
               '5_MLP-512': (114720.0, 230080.0), '5_MLP-1024': (229408.0, 459968.0),
               '5_ElementWise-sum': (2080.0, 4288.0), '5_ElementWise-avg': (2112.0, 4288.0),
               '5_ElementWise-min': (2080.0, 4288.0), '5_ElementWise-max': (2080.0, 4288.0),
               '5_ElementWise-innerproduct': (2272.0, 4288.0), '5_SelfAttention-1': (8000.0, 5344.0),
               '5_SelfAttention-2': (15968.0, 10496.0), '5_SelfAttention-3': (23936.0, 15648.0),
               '5_SelfAttention-4': (31904.0, 20800.0), '6_MLP-16': (4128.0, 8400.0),
               '6_MLP-32': (8224.0, 16608.0), '6_MLP-64': (16416.0, 33024.0), '6_MLP-128': (32800.0, 65856.0),
               '6_MLP-256': (65568.0, 131520.0), '6_MLP-512': (131104.0, 262848.0),
               '6_MLP-1024': (262176.0, 525504.0), '6_ElementWise-sum': (2080.0, 4288.0),
               '6_ElementWise-avg': (2112.0, 4288.0), '6_ElementWise-min': (2080.0, 4288.0),
               '6_ElementWise-max': (2080.0, 4288.0), '6_ElementWise-innerproduct': (2304.0, 4288.0),
               '6_SelfAttention-1': (9384.0, 5856.0), '6_SelfAttention-2': (18736.0, 11520.0),
               '6_SelfAttention-3': (28088.0, 17184.0), '6_SelfAttention-4': (37440.0, 22848.0),
               '7_MLP-16': (4640.0, 9424.0), '7_MLP-32': (9248.0, 18656.0), '7_MLP-64': (18464.0, 37120.0),
               '7_MLP-128': (36896.0, 74048.0), '7_MLP-256': (73760.0, 147904.0),
               '7_MLP-512': (147488.0, 295616.0), '7_MLP-1024': (294944.0, 591040.0),
               '7_ElementWise-sum': (2080.0, 4288.0), '7_ElementWise-avg': (2112.0, 4288.0),
               '7_ElementWise-min': (2080.0, 4288.0), '7_ElementWise-max': (2080.0, 4288.0),
               '7_ElementWise-innerproduct': (2336.0, 4288.0), '7_SelfAttention-1': (10784.0, 6368.0),
               '7_SelfAttention-2': (21536.0, 12544.0), '7_SelfAttention-3': (32288.0, 18720.0),
               '7_SelfAttention-4': (43040.0, 24896.0), '8_MLP-16': (5152.0, 10448.0),
               '8_MLP-32': (10272.0, 20704.0), '8_MLP-64': (20512.0, 41216.0), '8_MLP-128': (40992.0, 82240.0),
               '8_MLP-256': (81952.0, 164288.0), '8_MLP-512': (163872.0, 328384.0),
               '8_MLP-1024': (327712.0, 656576.0), '8_ElementWise-sum': (2080.0, 4288.0),
               '8_ElementWise-avg': (2112.0, 4288.0), '8_ElementWise-min': (2080.0, 4288.0),
               '8_ElementWise-max': (2080.0, 4288.0), '8_ElementWise-innerproduct': (2368.0, 4288.0),
               '8_SelfAttention-1': (12200.0, 6880.0), '8_SelfAttention-2': (24368.0, 13568.0),
               '8_SelfAttention-3': (36536.0, 20256.0), '8_SelfAttention-4': (48704.0, 26944.0),
               '9_MLP-16': (5632.0, 11472.0), '9_MLP-32': (11264.0, 22752.0), '9_MLP-64': (22528.0, 45312.0),
               '9_MLP-128': (45056.0, 90432.0), '9_MLP-256': (90112.0, 180672.0),
               '9_MLP-512': (180224.0, 361152.0), '9_MLP-1024': (360448.0, 722112.0),
               '9_ElementWise-sum': (2048.0, 4288.0), '9_ElementWise-avg': (2080.0, 4288.0),
               '9_ElementWise-min': (2048.0, 4288.0), '9_ElementWise-max': (2048.0, 4288.0),
               '9_ElementWise-innerproduct': (2368.0, 4288.0), '9_SelfAttention-1': (13600.0, 7392.0),
               '9_SelfAttention-2': (27200.0, 14592.0), '9_SelfAttention-3': (40800.0, 21792.0),
               '9_SelfAttention-4': (54400.0, 28992.0)}


class Block(nn.Module):
    """
    The input shape of raw sparse feature is (batch_size, field_size, embedding_dim).
    The input shape of raw dense feature is (batch_size, field_size, embedding_dim).
    The input shape of inner block is (batch_size, features_size).
    """

    def __init__(self, block_name, params, use_batchnorm=True, use_relu=True, use_dropout=False, dropout_rate=0,
                 use_linear=True):
        super(Block, self).__init__()
        self._block_name = block_name
        self._block_in_dim = params['block_in_dim']
        self._block_out_dim = params['block_out_dim']
        self._embedding_dim = params['embedding_dim']
        self._num_inputs = params['num_inputs']
        self._is_first_layer = params['is_first_layer']
        self._is_last_layer = params['is_last_layer']
        if self._is_last_layer:
            self._block_out_dim = params['tower_out_dim']
        if not self._is_last_layer:
            self._use_batchnorm = use_batchnorm
            # self._use_batchnorm = True
            self._use_relu = use_relu
            # TODO Attention!!!
            self._use_dropout = False
            # self._use_dropout = use_dropout
            self._dropout_rate = 0
            # self._dropout_rate = dropout_rate
        else:
            self._use_batchnorm = False
            self._use_relu = True
            self._use_dropout = False
            self._dropout_rate = 0

        self._relu = nn.ReLU()
        self._batchnorm = nn.BatchNorm1d(self._block_out_dim)
        self._dropout = nn.Dropout(self._dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: list, e.g. [(x1, input_type1), (x2, input_type2)]
        input_type == 0 means empty
        input_type == 1 means raw dense features
        input_type == 2 means raw sparse features
        input_type == 3 means inner block output features
        """
        raise NotImplementedError

    def get_flops(self):
        return 0

    @property
    def name(self):
        return self._block_name

    @property
    def num_params(self):
        return self._num_params

    def _count_params(self):
        return sum([p.numel() for p in self.parameters() if p is not None and p.requires_grad])


class MLP(Block):
    """
    This block applies MLP. The 3-D array will be converted into 3-D array.
    """

    def __init__(self, block_name, hidden_size, params, use_batchnorm=True, use_relu=True, use_dropout=False,
                 dropout_rate=0):
        super(MLP, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                  use_dropout=use_dropout, dropout_rate=dropout_rate)
        self._hidden_size = hidden_size
        # self._expand_linear = nn.Linear(self._block_in_dim, self._block_in_dim * self._num_inputs)
        in_dim = self._block_in_dim * self._num_inputs
        self._hidden_linear = nn.Linear(in_dim, hidden_size)
        self._output_linear = nn.Linear(hidden_size, self._block_out_dim)

        self._num_params = self._count_params()

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)

        final = self._hidden_linear(x)
        output = self._output_linear(final)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class ElementWise(Block):
    """
    This block applies inner product. The 3-D array will be converted into 3-D array.
    The elementwise type should be avg, sum, min, max or innerproduct.
    """

    def __init__(self, block_name, elementwise_type, params, use_batchnorm=True, use_relu=True, use_dropout=False,
                 dropout_rate=0):
        super(ElementWise, self).__init__(block_name, params,
                                          use_batchnorm=use_batchnorm, use_relu=use_relu, use_dropout=use_dropout,
                                          dropout_rate=dropout_rate)
        self._elementwise_type = elementwise_type
        # if self._is_first_layer:
        #     in_dim = self._embedding_dim
        #     self._output_linear = nn.Linear(in_dim, self._block_out_dim)
        # elif self._is_last_layer:
        #     in_dim = self._block_in_dim
        #     self._output_linear = nn.Linear(in_dim, self._block_out_dim)
        in_dim = self._block_in_dim
        self._output_linear = nn.Linear(in_dim, self._block_out_dim)
        # self._expand_linear = nn.Linear(self._block_in_dim, self._block_in_dim * self._num_inputs)

        self._num_params = self._count_params()

    def get_flops(self):
        if self._elementwise_type in ['sum', 'innerproduct']:
            return (self._num_inputs - 1) * self._block_in_dim
        elif self._elementwise_type in ['avg']:
            return self._num_inputs * self._block_in_dim
        else:
            return 0

    def forward(self, inputs):
        x = torch.stack(inputs, dim=0)

        if self._elementwise_type == 'avg':
            final = torch.mean(x, dim=0)
        elif self._elementwise_type == 'sum':
            final = torch.sum(x, dim=0)
        elif self._elementwise_type == 'min':
            final, _ = torch.min(x, dim=0)
        elif self._elementwise_type == 'max':
            final, _ = torch.max(x, dim=0)
        elif self._elementwise_type == 'innerproduct':
            final = torch.prod(x, dim=0)
        else:
            final = torch.sum(x, dim=0)

        # output = self._output_linear(final)
        output = final

        output = self._output_linear(output)

        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output


class SelfAttention(Block):
    def __init__(self, block_name, head_num, params, att_emb_dim=8,
                 use_residual=True, use_batchnorm=True, use_relu=True, use_dropout=False, dropout_rate=0):
        super(SelfAttention, self).__init__(block_name, params, use_batchnorm=use_batchnorm, use_relu=use_relu,
                                            use_dropout=use_dropout, dropout_rate=dropout_rate)
        self._head_num = head_num
        self._use_residual = use_residual
        self._att_emb_dim = att_emb_dim
        in_dim = self._block_in_dim
        self._query = nn.ModuleList([nn.Linear(in_dim, self._att_emb_dim) for _ in range(self._head_num)])
        self._key = nn.ModuleList([nn.Linear(in_dim, self._att_emb_dim) for _ in range(self._head_num)])
        self._value = nn.ModuleList([nn.Linear(in_dim, self._att_emb_dim) for _ in range(self._head_num)])

        if self._use_residual:
            self._residual = nn.Linear(in_dim, self._att_emb_dim * self._head_num)

        in_dim = self._att_emb_dim * self._head_num * self._num_inputs
        self._output_linear = nn.Linear(in_dim, self._block_out_dim)

        self._num_params = self._count_params()

    def _scaled_dot_product_attention(self, q, k, v):
        qk = torch.bmm(q, k.transpose(1, 2))
        dk = k.shape[-1]
        logits = qk / math.sqrt(dk)
        attention_weights = torch.softmax(logits, dim=-1)
        output = torch.bmm(attention_weights, v)

        return output

    def get_flops(self):
        return self._head_num * (
                    self._num_inputs * self._att_emb_dim * self._num_inputs * 2 + self._num_inputs * self._num_inputs * self._att_emb_dim * 2)

    def forward(self, inputs):
        x = torch.stack(inputs, dim=1)
        outputs = []
        for i in range(self._head_num):
            head = self._scaled_dot_product_attention(self._query[i](x), self._key[i](x), self._value[i](x))
            outputs.append(head)
        output = torch.cat(outputs, dim=2)
        if self._use_residual:
            output += self._residual(x)
        output = output.view(output.shape[0], -1)
        output = self._output_linear(output)
        if self._use_batchnorm:
            output = self._batchnorm(output)
        if self._use_relu:
            output = self._relu(output)
        if self._use_dropout:
            output = self._dropout(output)

        return output
