# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import torch
import numpy as np


def _lookup(x, vocab):
    x = x.tolist()
    y = []

    for _, batch in enumerate(x):
        ids = []
        for _, v in enumerate(batch):
            ids.append(vocab[v] if v in vocab else 2)
        y.append(ids)

    return torch.LongTensor(np.array(y, dtype="int32")).cuda()


def load_vocabulary(filename):
    vocab = []
    with open(filename, "rb") as fd:
        for line in fd:
            vocab.append(line.strip())

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return vocab, word2idx, idx2word


def load_tagged_vocabulary(filenames):
    vocab1 = []
    with open(filenames[0], "r") as f:
        for line in f:
            vocab1.append(line.strip())
    vocab2 = []
    with open(filenames[1], "r") as f:
        for line in f:
            vocab2.append(line.strip())

    src_word2idx = {}
    src_idx2word = {}
    tgt_word2idx = {}
    tgt_idx2word = {}
    for idx, word in enumerate(vocab1):
        src_word2idx[word] = idx
        src_idx2word[idx] = word
    for idx, word in enumerate(vocab2):
        tgt_word2idx[word] = idx
        tgt_idx2word[idx] = word

    src_tag_id_dict = {}  # >= 0
    prog = re.compile(r"<(/)?(\w*)>")
    id_cnt = 0
    tag_id_dict = {}
    src_tag_type_dict = {} # 0=None, -1=start, 1=end
    for i, token in enumerate(vocab1):
        match_obj = prog.match(token)
        if match_obj:
            if not token in tag_id_dict:
                tag_id_dict[token] = id_cnt
                id_cnt = id_cnt + 1
            if match_obj.group(1) == "/":
                src_tag_type_dict[i] = 1
            else:
                src_tag_type_dict[i] = -1
        else:
            src_tag_type_dict[i] = 0
    tgt_tag_type_dict = {} # 0=None, -1=start, 1=end
    for i, token in enumerate(vocab2):
        match_obj = prog.match(token)
        if match_obj:
            # assert the tags are the same
            if match_obj.group(1) == "/":
                tgt_tag_type_dict[i] = 1
            else:
                tgt_tag_type_dict[i] = -1
        else:
            tgt_tag_type_dict[i] = 0

    src_tag_id_dict = {}
    tgt_tag_id_dict = {}
    for i, token in enumerate(vocab1):
        if token in tag_id_dict:
            src_tag_id_dict[i] = tag_id_dict[token]
        else:
            src_tag_id_dict[i] = -1
    for i, token in enumerate(vocab2):
        if token in tag_id_dict:
            tgt_tag_id_dict[i] = tag_id_dict[token]
        else:
            tgt_tag_id_dict[i] = -1

    return {
        "vocab": vocab1,
        "word2idx": src_word2idx,
        "idx2word": src_idx2word,
        "tag_type": src_tag_type_dict,
        "tag_content": src_tag_id_dict,
    }, {
        "vocab": vocab2,
        "word2idx": tgt_word2idx,
        "idx2word": tgt_idx2word,
        "tag_type": tgt_tag_type_dict,
        "tag_content": tgt_tag_id_dict,
    }


def lookup(inputs, mode, params):
    if mode != "infer":
        features, labels = inputs
        source, target = features["source"], features["target"]
        source = source.numpy()
        target = target.numpy()
        labels = labels.numpy()
        src_mask = torch.FloatTensor(features["source_mask"].numpy()).cuda()
        tgt_mask = torch.FloatTensor(features["target_mask"].numpy()).cuda()
        enc_self_attn = torch.LongTensor(features["enc_self_attn"].numpy()).cuda()
        dec_self_attn = torch.LongTensor(features["dec_self_attn"].numpy()).cuda()
        enc_dec_attn = torch.LongTensor(features["enc_dec_attn"].numpy()).cuda()

        source = _lookup(source, params.lookup["source"])
        target = _lookup(target, params.lookup["target"])
        labels = _lookup(labels, params.lookup["target"])

        features = {
            "source": source,
            "source_mask": src_mask,
            "target": target,
            "target_mask": tgt_mask,
            "enc_self_attn": enc_self_attn,
            "dec_self_attn": dec_self_attn,
            "enc_dec_attn": enc_dec_attn,
        }

        return features, labels
    else:
        source = inputs["source"].numpy()
        source = _lookup(source, params.lookup["source"])
        src_mask = torch.FloatTensor(inputs["source_mask"].numpy()).cuda()
        enc_self_attn = torch.LongTensor(inputs["enc_self_attn"].numpy()).cuda()

        features = {
            "source": source,
            "source_mask": src_mask,
            "enc_self_attn": enc_self_attn,
        }

        return features
