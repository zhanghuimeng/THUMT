# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.distributed as dist
import thumt.utils as utils

from thumt.modules.module import Module
from thumt.modules.affine import Affine
from thumt.utils.helper import gen_typed_matrix_cpu, gen_typed_matrix_batch


class Attention(Module):

    def __init__(self, q_size, k_size, hidden_size, name="attention"):
        super(Attention, self).__init__(name)

        self._q_size = q_size
        self._k_size = k_size
        self._hidden_size = hidden_size

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size, name="q_transform")
            self.k_transform = Affine(k_size, hidden_size, name="k_transform")
            self.v_transform = Affine(hidden_size, 1,
                                      name="v_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, 1]
        logits = torch.transpose(logits, 1, 2)
        # [batch, 1, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.softmax(logits, dim=-1)

        # [batch, 1, length]
        weights = torch.squeeze(weights, 2)
        output = torch.matmul(weights, memory)

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight)
            nn.init.xavier_uniform_(self.k_transform.weight)
            nn.init.xavier_uniform_(self.v_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAttentionBase(Module):

    def __init__(self, name="multihead_attention_base"):
        super(MultiHeadAttentionBase, self).__init__(name=name)

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = [Affine(hidden_size, hidden_size,
                                      name="q_transform_%d" % i) for i in range(3)]
            self.k_transform = [Affine(hidden_size, hidden_size,
                                      name="k_transform_%d" % i) for i in range(3)]
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")

        self.reset_parameters()

    def forward(self, query, bias, seq_q, seq_k, vocab_q, vocab_k,
                memory=None, kv=None, attn_mat=None, mode="train", type="enc-self-attn"):
        # q shape (past): [batch, len_q, hidden]
        # q = self.q_transform(query)
        # q shape (now): [3, batch, len_q, hidden]
        q = torch.stack([self.q_transform[i](query) for i in range(3)])

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            # k = k or self.k_transform(memory)
            k = k or torch.stack([self.k_transform[i](query) for i in range(3)])
            v = v or self.v_transform(memory)
        else:
            # self-attention
            # k = self.k_transform(query)
            k = torch.stack([self.k_transform[i](query) for i in range(3)])
            v = self.v_transform(query)

            if kv is not None:
                # 问题：是否需要因为k的维度变化修改dim？
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # typed-attention matrix
        # typed_matrix: [3, batch, length_q, length_k]
        # or: [3, batch, 1, length_k]
        # or: [3, batch, 1, 1]
        if attn_mat is not None:
            typed_matrix = []
            for i in range(3):
                typed_matrix.append((attn_mat == i).float())
            # typed_matrix: [3, batch, length_q, length_k]
            typed_matrix = torch.stack(typed_matrix)
            # no need to patch for infer, already patched
        else:
            raise RuntimeError("there is no attn_mat")

        # split heads
        # because the shape of q is different, change split_heads
        # qh: [3, batch, heads, len_q, h2] or [3, batch, heads, 1, h2]
        # kh: [3, batch, heads, len_q, h2]
        # vh: [batch, heads, len_k, h2]
        # qh = self.split_heads(q, self.num_heads)
        # kh = self.split_heads(k, self.num_heads)
        qh = self.split_heads_4d(q, self.num_heads)
        kh = self.split_heads_4d(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention (removed)
        # kh = torch.transpose(kh, -2, -1)
        # logits = torch.matmul(qh, kh)

        # typed attention
        # kh: [3, batch, heads, h2, leng_k]
        kh = torch.transpose(kh, -2, -1)

        # # typed_attention
        # # us_weight: [3, 1, heads, h2, h2]
        # us_weight = torch.unsqueeze(self.typed_weight, 1)
        # # kh: [batch, heads, h2, length_k]
        # kh = torch.transpose(kh, -2, -1)
        # # typed_logits: [3, batch, heads, lq, lk]
        # typed_logits = torch.matmul(torch.matmul(qh, us_weight), kh)
        # typed_logits = typed_logits * typed_matrix.unsqueeze(2)
        # # logits: [batch, heads, lq, lk]
        # logits = torch.sum(typed_logits, dim=0)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    # x shape: [3, batch, len, hidden]
    @staticmethod
    def split_heads_4d(x, heads):
        batch = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.reshape(x, [x.shape[0], batch, length, heads, channels // heads])
        return torch.transpose(y, 3, 2)

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            for i in range(3):
                nn.init.xavier_uniform_(self.q_transform[i].weight, 2 ** -0.5)
                nn.init.xavier_uniform_(self.k_transform[i].weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            for i in range(3):
                nn.init.constant_(self.q_transform[i].bias, 0.0)
                nn.init.constant_(self.k_transform[i].bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAdditiveAttention(MultiHeadAttentionBase):

    def __init__(self, q_size, k_size, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAdditiveAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(k_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, num_heads,
                                      name="v_transform")
            self.o_transform = Affine(k_size, k_size,
                                      name="o_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, num_heads]
        logits = torch.transpose(logits, 1, 2)
        # [batch, num_heads, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        vh = self.split_heads(memory, self.num_heads)
        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)
