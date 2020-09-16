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
    tag_type_str = []  # std, start, end
    tag_type_id = []  # 0, -1, 1
    tag_content_str = []  # "", a
    tag_content_id = []  # ids
    tag_content_dict = {}

    def get_id(i, tag):
        if tag in tag_content_dict:
            return tag_content_dict[tag]
        tag_content_dict[tag] = i
        return i

    with open(filename, "r") as fd:
        for i, line in enumerate(fd):
            vocab.append(line.strip())
            if vocab[-1] == params.pad or vocab[-1] == params.bos or vocab[-1] == params.eos or vocab[-1] == params.unk:
                tag_type_str.append("std")
                tag_type_id.append(0)
                tag_content_str.append("")
                tag_content_id.append(-1)
                continue
            match_obj = re.match(r"<(/)?(\w*)>", vocab[-1])
            if match_obj:
                tag_content_str.append(match_obj.group(2))
                if match_obj.group(1) == "/":
                    tag_type_str.append("end")
                    tag_type_id.append(1)
                    tag_content_id.append(get_id(i, tag_content_str[-1]))
                else:
                    tag_type_str.append("start")
                    tag_type_id.append(-1)
                    tag_content_id.append(get_id(i, tag_content_str[-1]))
            else:
                tag_type_str.append("std")
                tag_type_id.append(0)
                tag_content_str.append("")
                tag_content_id.append(-1)

            # if match_obj:
            #     print("vocab=%s tagtype=(%s, %d) tagcontent=(%s, %d)" %
            #           (vocab[-1], tag_type_str[-1], tag_type_id[-1], tag_content_str[-1], tag_content_id[-1]))

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return {
        "vocab": vocab,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "tag_type_str": tag_type_str,
        "tag_content_str": tag_content_str,
        "tag_attr": torch.transpose(torch.LongTensor([tag_type_id, tag_content_id]), 1, 0)
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
