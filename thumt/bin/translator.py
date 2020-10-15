# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import six
import socket
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cProfile
import pstats
import io

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.utils as utils

plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--log_dir", type=str, required=True,
                        default="test/",
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, required=True, nargs="+",
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")
    parser.add_argument("--save_attn_fig", action="store_true",
                        help="Save typed-attn figure for each sentence")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device_list=[0],
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        decode_batch_size=32,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters.lower())

    src_vocabulary, tgt_vocabulary = data.load_tagged_vocabulary(args.vocabulary)
    params.vocab = args.vocabulary
    params.full_vocab = src_vocabulary, tgt_vocabulary

    params.vocabulary = {
        "source": src_vocabulary["vocab"], "target": tgt_vocabulary["vocab"]
    }
    params.lookup = {
        "source": src_vocabulary["word2idx"], "target": tgt_vocabulary["word2idx"]
    }
    params.mapping = {
        "source": src_vocabulary["idx2word"], "target": tgt_vocabulary["idx2word"]
    }

    return params


def convert_to_string(tensor, params):
    ids = tensor.tolist()

    output = []

    for wid in ids:
        if wid == 1:
            break
        output.append(params.mapping["target"][wid])

    output = b" ".join(output)

    return output


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def main(args):
    # cProfile
    pr = cProfile.Profile()
    pr.enable()

    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_params() for _ in range(len(model_cls_list))]
    params_list = [
        merge_params(params, model_cls.default_params())
        for params, model_cls in zip(params_list, model_cls_list)]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))]
    params_list = [
        override_params(params_list[i], args)
        for i in range(len(model_cls_list))]

    params = params_list[0]
    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Create model
    with torch.no_grad():
        model_list = []

        for i in range(len(args.models)):
            model = model_cls_list[i](params_list[i]).cuda()

            if args.half:
                model = model.half()

            model.eval()
            model.load_state_dict(
                torch.load(args.checkpoints[i],
                           map_location="cpu")["model"])

            model_list.append(model)

        if len(args.input) == 2:
            mode = "infer"
            sorted_key, sorted_data, dataset = data.get_dataset(
                args.input, mode, params)
        else:
            # Teacher-forcing
            mode = "eval"
            dataset = data.get_dataset(args.input, mode, params)
            sorted_key = None

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024
        top_beams = params.top_beams
        decode_batch_size = params.decode_batch_size

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([decode_batch_size, top_beams, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        
        all_outputs = []
        all_dec_self_attn = []
        all_enc_dec_attn = []

        start_time = time.time()
        if args.log_dir is not None:
            writer = torch.utils.tensorboard.SummaryWriter(args.log_dir)
        else:
            writer = None
        while True:
            try:
                features = next(iterator)
                features = data.lookup(features, mode, params)

                if mode == "eval":
                    features = features[0]

                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float(),
                    "enc_self_attn": torch.ones([1, 1, 1]).float()
                }

                if mode == "eval":
                    features["target"] = torch.ones([1, 1]).long()
                    features["target_mask"] = torch.ones([1, 1]).float()

                batch_size = 0

            t = time.time()
            counter += 1

            # Decode
            if mode != "eval":
                seqs, _ = utils.beam_search(model_list, features, params, counter, writer)
            else:
                seqs, _ = utils.argmax_decoding(model_list, features, params)

            # Padding
            pad_batch = decode_batch_size - seqs.shape[0]
            pad_beams = top_beams - seqs.shape[1]
            pad_length = pad_max - seqs.shape[2]
            seqs = torch.nn.functional.pad(
                seqs, (0, pad_length, 0, pad_beams, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(decode_batch_size):
                for j in range(dist.get_world_size()):
                    beam_seqs = []
                    pad_flag = i >= size[j]
                    for k in range(top_beams):
                        seq = convert_to_string(t_list[j][i][k], params)

                        if pad_flag:
                            continue
                        
                        beam_seqs.append(seq)
                    
                    if pad_flag:
                        continue
                    
                    all_outputs.append(beam_seqs)

            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))

        if writer:
            writer.close()
        print("Total time: %.3f sec" % (time.time() - start_time))

        if dist.get_rank() == 0:
            restored_inputs = []
            restored_outputs = []
            if sorted_key is not None:
                for idx in range(len(all_outputs)):
                    restored_inputs.append(sorted_data[sorted_key[idx]])
                    restored_outputs.append(all_outputs[sorted_key[idx]])
            else:
                restored_outputs = all_outputs
            
            with open(args.output, "wb") as fd:
                if top_beams == 1:
                    for seqs in restored_outputs:
                        fd.write(seqs[0] + b"\n")
                else:
                    for idx, seqs in enumerate(restored_outputs):
                        for k, seq in enumerate(seqs):
                            fd.write(b"%d\t%d\t" % (idx, k))
                            fd.write(seq + b"\n")

    # end of cProfile
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open(os.path.join(args.log_dir, "profile.txt"), "w") as f:
        f.write(s.getvalue())


# 0: green, 1: red, 2: blue
colors = ["green", "red", "blue"]
cmap = ListedColormap(colors)

def save_attn_fig(filename, seq_q, seq_k, mat):
    nq = len(seq_q)
    nk = len(seq_k)
    extent = (0, nk, nq, 0)
    _, ax = plt.subplots(figsize=((nk + 3) // 4 + 4, (nq + 3) // 4 + 4))
    ax.imshow(mat, vmin=0, vmax=len(cmap.colors), cmap=cmap, extent=extent)
    ax.set_frame_on(False)
    locs = np.arange(nk)
    ax.xaxis.set_ticks(locs, minor=True)
    ax.xaxis.set(ticks=locs + 0.5, ticklabels=seq_k)
    locs = np.arange(nq)
    ax.yaxis.set_ticks(locs, minor=True)
    ax.yaxis.set(ticks=locs + 0.5, ticklabels=seq_q)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.grid(color='w', linewidth=1, which="minor")
    plt.savefig(filename)
    plt.close()


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
