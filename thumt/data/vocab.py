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


def load_tagged_vocabulary(params, filename):
    vocab = []
    tagtype = []
    tagcontent = []
    with open(filename, "r") as fd:
        for line in fd:
            vocab.append(line.strip())
            if vocab[-1] == params.pad or vocab[-1] == params.bos or vocab[-1] == params.eos or vocab[-1] == params.unk:
                tagtype.append("std")
                tagcontent.append("")
                continue
            match_obj = re.match(r"<(/)?(\w*)>", vocab[-1])
            if match_obj:
                if match_obj.group(1) == "/":
                    tagtype.append("end")
                else:
                    tagtype.append("start")
                tagcontent.append(match_obj.group(2))
            else:
                tagtype.append("std")
                tagcontent.append("")

            # if match_obj:
            #     print("vocab=%s tagtype=%s tagcontent=%s" % (vocab[-1], tagtype[-1], tagcontent[-1]))

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return vocab, word2idx, idx2word, tagtype, tagcontent


def lookup(inputs, mode, params):
    if mode != "infer":
        features, labels = inputs
        source, target = features["source"], features["target"]
        source = source.numpy()
        target = target.numpy()
        labels = labels.numpy()
        src_mask = torch.FloatTensor(features["source_mask"].numpy()).cuda()
        tgt_mask = torch.FloatTensor(features["target_mask"].numpy()).cuda()

        source = _lookup(source, params.lookup["source"])
        target = _lookup(target, params.lookup["target"])
        labels = _lookup(labels, params.lookup["target"])

        features = {
            "source": source,
            "source_mask": src_mask,
            "target": target,
            "target_mask": tgt_mask
        }

        return features, labels
    else:
        source = inputs["source"].numpy()
        source = _lookup(source, params.lookup["source"])
        src_mask = torch.FloatTensor(inputs["source_mask"].numpy()).cuda()

        features = {
            "source": source,
            "source_mask": src_mask
        }

        return features
