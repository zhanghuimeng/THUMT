# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import tensorflow as tf
from time import time as python_time

from collections import namedtuple
from thumt.utils.nest import map_structure
import thumt.data as data
import thumt.utils.helper as helper


def _merge_first_two_dims(tensor):
    if isinstance(tensor, torch.Tensor):
        shape = list(tensor.shape)
        shape[1] *= shape[0]
        return torch.reshape(tensor, shape[1:])
    elif isinstance(tensor, np.ndarray):
        shape = list(tensor.shape)
        shape[1] *= shape[0]
        return np.reshape(tensor, shape[1:])
    else:
        raise ValueError("Unknown tensor type")


def _split_first_two_dims(tensor, dim_0, dim_1):
    if isinstance(tensor, torch.Tensor):
        shape = [dim_0, dim_1] + list(tensor.shape)[1:]
        return torch.reshape(tensor, shape)
    elif isinstance(tensor, np.ndarray):
        shape = [dim_0, dim_1] + list(tensor.shape)[1:]
        return np.reshape(tensor, shape)
    else:
        raise ValueError("Unknown tensor type")


def _tile_to_beam_size(tensor, beam_size):
    if isinstance(tensor, torch.Tensor):
        tensor = torch.unsqueeze(tensor, 1)
        tile_dims = [1] * int(tensor.dim())
        tile_dims[1] = beam_size
        return tensor.repeat(tile_dims)
    elif isinstance(tensor, np.ndarray):
        tensor = np.expand_dims(tensor, 1)
        tile_dims = [1] * int(tensor.ndim)
        tile_dims[1] = beam_size
        return np.tile(tensor, tile_dims)
    else:
        raise ValueError("Unknown tensor type")


def _gather_2d(params, indices, name=None):
    if isinstance(params, torch.Tensor):
        batch_size = params.shape[0]
        range_size = indices.shape[1]
        batch_pos = torch.arange(batch_size * range_size, device=params.device)
        batch_pos = batch_pos // range_size
        batch_pos = torch.reshape(batch_pos, [batch_size, range_size])
        output = params[batch_pos, indices]

        return output
    elif isinstance(params, np.ndarray):
        batch_size = params.shape[0]
        range_size = indices.shape[1]
        batch_pos = np.arange(batch_size * range_size)
        batch_pos = batch_pos // range_size
        batch_pos = np.reshape(batch_pos, [batch_size, range_size])
        output = params[batch_pos, indices.cpu().numpy()]
        return output
    else:
        raise ValueError("Unknown tensor type")


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
    pass


def _get_inference_fn(model_fns, features, to_cpu=False):
    def inference_fn(inputs, state, step):
        length_k = features["source"].shape[-1]
        dec_self_attn = torch.from_numpy(
            state[0]["typed_matrix"]["dec_self_attn"]["mat"][:, step, np.newaxis, :step + 1])
        enc_dec_attn = torch.from_numpy(
            state[0]["typed_matrix"]["enc_dec_attn"]["mat"][:, step, np.newaxis, :length_k])
        if not to_cpu:
            dec_self_attn = dec_self_attn.cuda()
            enc_dec_attn = enc_dec_attn.cuda()
        local_features = {
            "source": features["source"],
            "source_mask": features["source_mask"],
            "enc_self_attn": features["enc_self_attn"],
            "dec_self_attn": dec_self_attn,
            "enc_dec_attn": enc_dec_attn,
            "target": inputs,
            "target_mask": torch.ones(*inputs.shape).to(inputs).float()
        }

        outputs = []
        next_state = []

        for (model_fn, model_state) in zip(model_fns, state):
            if model_state:
                logits, new_state = model_fn(local_features, model_state)
                outputs.append(torch.nn.functional.log_softmax(logits,
                                                               dim=-1))
                next_state.append(new_state)
            else:
                logits = model_fn(local_features)
                outputs.append(torch.nn.functional.log_softmax(logits,
                                                               dim=-1))
                next_state.append({})

        # Ensemble
        log_prob = sum(outputs) / float(len(outputs))

        return log_prob.float(), next_state

    return inference_fn


def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id, min_length, max_length, inf=-1e9, vocab=None,
                      source=None, src_vocab=None,
                      length_src=None, epoch=-1, writer=None):
    time_point = python_time()
    # Compute log probabilities
    seqs, log_probs = state.inputs[:2]
    flat_seqs = _merge_first_two_dims(seqs)
    flat_state = map_structure(lambda x: _merge_first_two_dims(x), state.state)
    if writer:
        writer.add_scalar("beam_search/epoch_%d/initialize" % epoch,
                          python_time() - time_point, time)
    time_point = python_time()
    step_log_probs, next_state = func(flat_seqs, flat_state, time)
    if writer:
        writer.add_scalar("beam_search/epoch_%d/inference_fn" % epoch,
                          python_time() - time_point, time)
    time_point = python_time()
    step_log_probs = _split_first_two_dims(step_log_probs, batch_size,
                                           beam_size)
    next_state = map_structure(
        lambda x: _split_first_two_dims(x, batch_size, beam_size), next_state)
    curr_log_probs = torch.unsqueeze(log_probs, 2) + step_log_probs

    # Apply length penalty
    length_penalty = ((5.0 + float(time + 1)) / 6.0) ** alpha
    curr_scores = curr_log_probs / length_penalty
    vocab_size = curr_scores.shape[-1]

    # Prevent null translation
    min_length_flags = torch.ge(min_length, time + 1).float().mul_(inf)
    curr_scores[:, :, eos_id].add_(min_length_flags)

    # Select top-k candidates
    # [batch_size, beam_size * vocab_size]
    curr_scores = torch.reshape(curr_scores, [-1, beam_size * vocab_size])
    # [batch_size, 2 * beam_size]
    top_scores, top_indices = torch.topk(curr_scores, k=2*beam_size)
    # Shape: [batch_size, 2 * beam_size]
    beam_indices = top_indices // vocab_size
    symbol_indices = top_indices % vocab_size
    # Expand sequences
    # [batch_size, 2 * beam_size, time]
    candidate_seqs = _gather_2d(seqs, beam_indices)
    candidate_seqs = torch.cat([candidate_seqs,
                                torch.unsqueeze(symbol_indices, 2)], 2)

    # Expand sequences
    # Suppress finished sequences
    flags = torch.eq(symbol_indices, eos_id).to(torch.bool)
    # [batch, 2 * beam_size]
    alive_scores = top_scores + flags.to(torch.float32) * inf
    # [batch, beam_size]
    alive_scores, alive_indices = torch.topk(alive_scores, beam_size)
    alive_symbols = _gather_2d(symbol_indices, alive_indices)
    alive_indices = _gather_2d(beam_indices, alive_indices)
    alive_seqs = _gather_2d(seqs, alive_indices)
    # [batch_size, beam_size, time + 1]
    alive_seqs = torch.cat([alive_seqs, torch.unsqueeze(alive_symbols, 2)], 2)
    alive_state = map_structure(
        lambda x: _gather_2d(x, alive_indices),
        next_state)
    alive_log_probs = alive_scores * length_penalty
    # update the matrices
    alive_state = map_structure(
        lambda x: _merge_first_two_dims(x),
        alive_state
    )
    if writer:
        writer.add_scalar("beam_search/epoch_%d/beam_search_1" % epoch,
                          python_time() - time_point, time)
    time_point = python_time()
    tgt_seq = map_structure(
        lambda x: _merge_first_two_dims(x),
        alive_seqs,
    ).cpu().numpy()
    for model_state in alive_state:
        helper.update_tgt_stack_batch_cpu(
            step=time + 1,
            seq=np.reshape(alive_symbols.cpu().numpy(), [-1]),
            stack=model_state["typed_matrix"]["stack_q"],
            stack_pointer=model_state["typed_matrix"]["stack_pointer_q"],
            nearest_q=model_state["typed_matrix"]["nearest_q"],
            stack_history_k=model_state["typed_matrix"]["dec_self_attn"]["stack_history_k"],
            vocab=vocab,
        )
        helper.update_dec_self_attn_batch_cpu(
            step=time + 1,
            mat=model_state["typed_matrix"]["dec_self_attn"]["mat"],
            nearest_q=model_state["typed_matrix"]["nearest_q"],
            stack_history_k=model_state["typed_matrix"]["dec_self_attn"]["stack_history_k"],
        )
        helper.update_enc_dec_attn_batch_cpu(
            step=time + 1,
            length_k=length_src,
            mat=model_state["typed_matrix"]["enc_dec_attn"]["mat"],
            nearest_q=model_state["typed_matrix"]["nearest_q"],
            stack_history_k=model_state["typed_matrix"]["enc_dec_attn"]["stack_history_k"],
        )
        # print("step=%d" % (time + 1))
        # helper.print_state(
        #     step=time + 1,
        #     src_seq=source.cpu().numpy(),
        #     tgt_seq=tgt_seq,
        #     src_vocab=src_vocab,
        #     tgt_vocab=vocab,
        #     state=model_state
        # )
    alive_state = map_structure(
        lambda x: _split_first_two_dims(x, batch_size, beam_size),
        alive_state
    )
    if writer:
        writer.add_scalar("beam_search/epoch_%d/attn_mat" % epoch,
                          python_time() - time_point, time)
    time_point = python_time()

    # Check length constraint
    length_flags = torch.le(max_length, time + 1).float()
    alive_log_probs = alive_log_probs + length_flags * inf
    alive_scores = alive_scores + length_flags * inf

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
    # [batch, 2 * beam_size]
    step_fin_scores = top_scores + (1.0 - flags.to(torch.float32)) * inf
    # [batch, 3 * beam_size]
    fin_flags = torch.cat([prev_fin_flags, flags], dim=1)
    fin_scores = torch.cat([prev_fin_scores, step_fin_scores], dim=1)
    # [batch, beam_size]
    fin_scores, fin_indices = torch.topk(fin_scores, beam_size)
    fin_flags = _gather_2d(fin_flags, fin_indices)
    pad_seqs = prev_fin_seqs.new_full([batch_size, beam_size, 1], pad_id)
    prev_fin_seqs = torch.cat([prev_fin_seqs, pad_seqs], dim=2)
    fin_seqs = torch.cat([prev_fin_seqs, candidate_seqs], dim=1)
    fin_seqs = _gather_2d(fin_seqs, fin_indices)

    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )
    if writer:
        writer.add_scalar("beam_search/epoch_%d/beam_search_2" % epoch,
                          python_time() - time_point, time)

    return new_state


def beam_search(models, features, params, epoch=-1, writer=None, to_cpu=False):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha
    decode_ratio = params.decode_ratio
    decode_length = params.decode_length

    pad_id = params.lookup["target"][params.pad.encode("utf-8")]
    bos_id = params.lookup["target"][params.bos.encode("utf-8")]
    eos_id = params.lookup["target"][params.eos.encode("utf-8")]

    min_val = -1e9
    shape = features["source"].shape
    device = features["source"].device
    batch_size = shape[0]
    seq_length = shape[1]

    # Compute initial state if necessary
    states = []
    funcs = []

    for model in models:
        state = model.empty_state(batch_size, device)
        states.append(model.encode(features, state, "infer"))
        funcs.append(model.decode)

    # initialize the stacks in model
    src_vocabulary, tgt_vocabulary = params.full_vocab
    seq_k = features["source"].cpu().numpy()
    for state in states:
        # initialize enc_dec_attn stack_history
        stack_history_k = state["typed_matrix"]["enc_dec_attn"]["stack_history_k"]
        helper.calc_stack_history_batch_cpu(
            seq=seq_k,
            stack_history=stack_history_k,
            vocab=src_vocabulary
        )

    # For source sequence length
    max_length = features["source_mask"].sum(1) * decode_ratio
    max_length = max_length.long() + decode_length
    max_step = max_length.max()
    # [batch, beam_size]
    max_length = torch.unsqueeze(max_length, 1).repeat([1, beam_size])
    min_length = torch.ones_like(max_length)

    # Expand the inputs
    # [batch, length] => [batch * beam_size, length]
    features["source"] = torch.unsqueeze(features["source"], 1)
    features["source"] = features["source"].repeat([1, beam_size, 1])
    features["source"] = torch.reshape(features["source"],
                                       [batch_size * beam_size, seq_length])
    features["source_mask"] = torch.unsqueeze(features["source_mask"], 1)
    features["source_mask"] = features["source_mask"].repeat([1, beam_size, 1])
    features["source_mask"] = torch.reshape(features["source_mask"],
                                       [batch_size * beam_size, seq_length])
    # added enc_self_attn: [batch, lq, lq] => [batch * beam_size, lq, lq]
    features["enc_self_attn"] = torch.unsqueeze(features["enc_self_attn"], 1)
    features["enc_self_attn"] = features["enc_self_attn"].repeat([1, beam_size, 1, 1])
    features["enc_self_attn"] = torch.reshape(features["enc_self_attn"],
                                            [batch_size * beam_size, seq_length, seq_length])

    # fixed the way to modify
    decoding_fn = _get_inference_fn(funcs, features, to_cpu)

    states = map_structure(
        lambda x: _tile_to_beam_size(x, beam_size),
        states)

    # Initial beam search state
    init_seqs = torch.full([batch_size, beam_size, 1], bos_id, device=device)
    init_seqs = init_seqs.long()
    # initialize typed_matrix
    flat_states = map_structure(lambda x: _merge_first_two_dims(x), states)
    for state in flat_states:
        helper.update_tgt_stack_batch_cpu(
            step=0,
            seq=np.reshape(init_seqs.cpu().numpy(), [-1]),
            stack=state["typed_matrix"]["stack_q"],
            stack_pointer=state["typed_matrix"]["stack_pointer_q"],
            nearest_q=state["typed_matrix"]["nearest_q"],
            stack_history_k=state["typed_matrix"]["dec_self_attn"]["stack_history_k"],
            vocab=tgt_vocabulary,
        )
        helper.update_dec_self_attn_batch_cpu(
            step=0,
            mat=state["typed_matrix"]["dec_self_attn"]["mat"],
            nearest_q=state["typed_matrix"]["nearest_q"],
            stack_history_k=state["typed_matrix"]["dec_self_attn"]["stack_history_k"],
        )
        helper.update_enc_dec_attn_batch_cpu(
            step=0,
            length_k=features["source"].shape[-1],
            mat=state["typed_matrix"]["enc_dec_attn"]["mat"],
            nearest_q=state["typed_matrix"]["nearest_q"],
            stack_history_k=state["typed_matrix"]["enc_dec_attn"]["stack_history_k"],
        )
        # helper.print_state(0, features["source"].shape[-1], state)
    states = map_structure(lambda x: _split_first_two_dims(x, batch_size, beam_size), flat_states)

    init_log_probs = init_seqs.new_tensor(
        [[0.] + [min_val] * (beam_size - 1)], dtype=torch.float32)
    init_log_probs = init_log_probs.repeat([batch_size, 1])
    init_scores = torch.zeros_like(init_log_probs)
    fin_seqs = torch.zeros([batch_size, beam_size, 1], dtype=torch.int64,
                           device=device)
    fin_scores = torch.full([batch_size, beam_size], min_val,
                            dtype=torch.float32, device=device)
    fin_flags = torch.zeros([batch_size, beam_size], dtype=torch.bool,
                            device=device)

    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores),
        state=states,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    length_src = features["source"].shape[-1]
    for time in range(max_step):
        state = _beam_search_step(
            time=time, func=decoding_fn, state=state, batch_size=batch_size,
            beam_size=beam_size, alpha=alpha, pad_id=pad_id, eos_id=eos_id, min_length=min_length,
            max_length=max_length, inf=-1e9, vocab=tgt_vocabulary,
            source=features["source"], src_vocab=src_vocabulary,
            length_src=length_src, epoch=epoch, writer=writer)
        max_penalty = ((5.0 + max_step) / 6.0) ** alpha
        best_alive_score = torch.max(state.inputs[1][:, 0] / max_penalty)
        worst_finished_score = torch.min(state.finish[2])
        cond = torch.gt(worst_finished_score, best_alive_score)
        is_finished = bool(cond)

        if writer:
            writer.flush()

        if is_finished:
            break

    final_state = state
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0].byte()
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    final_seqs = torch.where(final_flags[:, :, None], final_seqs, alive_seqs)
    final_scores = torch.where(final_flags, final_scores, alive_scores)

    # Append extra <eos>
    final_seqs = torch.nn.functional.pad(final_seqs, (0, 1, 0, 0, 0, 0),
                                         value=eos_id)

    # with np.printoptions(threshold=np.inf):
    #     print(final_state.state[0]["typed_matrix"]["dec_self_attn"]["mat"].shape)
    #     print(final_state.state[0]["typed_matrix"]["dec_self_attn"]["mat"][:, :, :20, :20])
    #     print(final_state.state[0]["typed_matrix"]["enc_dec_attn"]["mat"].shape)
    #     print(final_state.state[0]["typed_matrix"]["enc_dec_attn"]["mat"][:, :, :20, :20])
    # [batch, beam, leng, leng]
    dec_self_attn = final_state.state[0]["typed_matrix"]["dec_self_attn"]["mat"][:, 0, :, :]
    enc_dec_attn = final_state.state[0]["typed_matrix"]["enc_dec_attn"]["mat"][:, 0, :, :]

    # with np.printoptions(threshold=np.inf):
    #     for i in range(beam_size):
    #         print("src:")
    #         helper.print_sentence(features["source"][i // 4], src_vocabulary["idx2word"])
    #         print("tgt:")
    #         helper.print_sentence(final_seqs[0, i, :], tgt_vocabulary["idx2word"])
    #         print("dec_self_attn: ")
    #         print(final_state.state[0]["typed_matrix"]["dec_self_attn"]["mat"][0, i, :20, :20])
    #         print("enc_dec_attn: ")
    #         print(final_state.state[0]["typed_matrix"]["enc_dec_attn"]["mat"][0, i, :20, :20])

    return final_seqs[:, :top_beams, 1:], final_scores[:, :top_beams], dec_self_attn, enc_dec_attn


def argmax_decoding(models, features, params):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    # Compute initial state if necessary
    log_probs = []
    shape = features["target"].shape
    device = features["target"].device
    batch_size = features["target"].shape[0]
    target_mask = features["target_mask"]
    target_length = target_mask.sum(1).long()
    eos_id = params.lookup["target"][params.eos.encode("utf-8")]

    for model in models:
        state = model.empty_state(batch_size, device)
        state = model.encode(features, state)
        logits, _ = model.decode(features, state, "eval")
        log_probs.append(torch.nn.functional.log_softmax(logits, dim=-1))

    log_prob = sum(log_probs) / len(models)
    ret = torch.max(log_prob, -1)
    values = torch.reshape(ret.values, shape)
    indices = torch.reshape(ret.indices, shape)

    batch_pos = torch.arange(batch_size, device=device)
    seq_pos = target_length - 1
    indices[batch_pos, seq_pos] = eos_id

    return indices[:, None, :], torch.sum(values * target_mask, -1)[:, None]
