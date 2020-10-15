import logging
import numpy as np
import torch


def print_sentence(sentence, idx2word):
    if isinstance(sentence, torch.Tensor):
        sentence = sentence.cpu().numpy()
    if sentence.ndim == 1:
        tokens = [idx2word[id] for id in sentence]
        print(b" ".join(tokens).decode("utf-8"))
    elif sentence.ndim == 2:
        if len(sentence) > 5:
            sentence = sentence[:5, :]
        for i in range(len(sentence)):
            tokens = [idx2word[id] for id in sentence[i]]
            print(b" ".join(tokens).decode("utf-8"))
    else:
        raise ValueError("Can't handle more than 2 dimensions")


def gen_typed_matrix_cpu(seq_q, seq_k, vocab_q, vocab_k):
    batch, nq = seq_q.shape
    _, nk = seq_k.shape
    nearest_q = [[-1 for _ in range(nq)] for _ in range(batch)]
    stack = [[] for _ in range(batch)]

    for i in range(batch):
        for j in range(nq):
            x = seq_q[i][j]
            if vocab_q["tag_type"][x] == -1:
                stack[i].append(vocab_q["tag_content"][x])
            if len(stack[i]) == 0:
                nearest_q[i][j] = -1
            else:
                nearest_q[i][j] = stack[i][-1]
            if vocab_q["tag_type"][x] == 1 and len(stack[i]) > 0:
                stack[i].pop()

    typed_matrix = np.zeros([3, batch, nq, nk])
    stack = [[] for _ in range(batch)]
    for i in range(batch):
        for j in range(nk):
            x = seq_k[i][j]
            if vocab_k["tag_type"][x] == -1:
                stack[i].append(vocab_k["tag_content"][x])
            for k in range(nq):
                # 0: std, 1: in, 2: out
                if nearest_q[i][k] == -1:
                    typed_matrix[0][i][k][j] = 1
                elif nearest_q[i][k] in stack[i]:
                    typed_matrix[1][i][k][j] = 1
                else:
                    typed_matrix[2][i][k][j] = 1
            if vocab_k["tag_type"][x] == 1 and len(stack[i]) > 0:
                stack[i].pop()

    return torch.from_numpy(typed_matrix).int().cuda()


def gen_typed_matrix_batch(seq_q, seq_k, vocab_q, vocab_k):
    batch, lq = seq_q.shape
    _, lk = seq_k.shape

    # gather batch information
    # [batch, lq, 2]
    tag_attr_q = torch.nn.functional.embedding(seq_q, vocab_q["tag_attr"].cuda(seq_q.get_device()))
    tag_attr_k = torch.nn.functional.embedding(seq_k, vocab_k["tag_attr"].cuda(seq_k.get_device()))
    # [batch, lq]
    tag_type_q = torch.squeeze(tag_attr_q[:, :, 0])
    tag_content_q = torch.squeeze(tag_attr_q[:, :, 1])
    tag_type_k = torch.squeeze(tag_attr_k[:, :, 0])
    tag_content_k = torch.squeeze(tag_attr_k[:, :, 1])

    def update_nearest_matrix(l, tag_type, tag_content):
        # print("tag_type: %s" % str(tag_type.size()))
        nearest = torch.zeros([batch, l]).long()
        stack = torch.full([batch, l], -1).long()
        stack_pointer = torch.ones([batch]).long()
        # stack operations
        for i in range(l):
            # indices to push and pop
            # [batch]
            open_tag_indices = tag_type[:, i] == -1
            close_tag_indices = tag_type[:, i] == 1
            # push values to stack
            stack[open_tag_indices, stack_pointer[open_tag_indices]] = tag_content[open_tag_indices, i]
            # increment stack pointer
            stack_pointer[open_tag_indices] = stack_pointer[open_tag_indices] + 1
            # update "nearest" matrix
            nearest[:, i] = stack.gather(1, (stack_pointer - 1).unsqueeze(1)).squeeze()
            # pop values from stack
            stack_pointer[close_tag_indices] = stack_pointer[close_tag_indices] - 1
        return nearest

    nearest_q = update_nearest_matrix(lq, tag_type_q, tag_content_q)
    nearest_k = update_nearest_matrix(lk, tag_type_k, tag_content_k)

    # for i in range(batch):
    #     print_sentence(seq_q[i], vocab_q["idx2word"])
    #     print(nearest_q[i])
    #     print_sentence(seq_k[i], vocab_k["idx2word"])
    #     print(nearest_k[i])
    #     print()

    # 0: std, 1: in, 2: out
    # typed_matrix = torch.zeros([3, batch, lq, lk])
    # for i in range(batch):
    #     for j in range(lq):
    #         for k in range(lk):
    #             if nearest_q[i][j] == -1:
    #                 typed_matrix[0][i][j][k] = 1
    #             elif nearest_q[i][j] == nearest_k[i][k]:
    #                 typed_matrix[1][i][j][k] = 1
    #             else:
    #                 typed_matrix[2][i][j][k] = 1
    typed_matrix_0 = (nearest_q == -1).long().unsqueeze(-1).repeat(1, 1, lk)
    typed_matrix_12 = nearest_q.unsqueeze(-1) - nearest_k.unsqueeze(-2)
    typed_matrix_1 = (typed_matrix_12 == 0).long()
    # mask out typed_matrix_0
    typed_matrix_1 = typed_matrix_1 * (1 - typed_matrix_0)
    typed_matrix_2 = (typed_matrix_12 != 0).long()
    typed_matrix_2 = typed_matrix_2 * (1 - typed_matrix_0)

    # need to check
    # check_matrix = typed_matrix_0 + typed_matrix_1 * 2 + typed_matrix_2 * 3
    # for i in range(batch):
    #     print_sentence(seq_q[i], vocab_q["idx2word"])
    #     print_sentence(seq_k[i], vocab_k["idx2word"])
    #     print(check_matrix[i])
    #     print()

    return torch.stack([typed_matrix_0, typed_matrix_1, typed_matrix_2])


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# vocab: dict
def stack_push_batch_cpu(seq, stack, pointer, vocab):
    batch_size = seq.shape[0]
    for i in range(batch_size):
        if vocab["tag_type"][seq[i]] == -1:
            stack[i][pointer[i]] = vocab["tag_content"][seq[i]]
            pointer[i] += 1


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# vocab: dict
def stack_pop_batch_cpu(seq, stack, pointer, vocab):
    batch_size = seq.shape[0]
    for i in range(batch_size):
        if vocab["tag_type"][seq[i]] == 1 and pointer[i] > 0:
            stack[i][pointer[i] - 1] = -1 # for stack_history
            pointer[i] -= 1


# calculate batches of stack history (used for src)
# seq: [batch, length]
# stack_history: [batch, max_length, max_length]
# vocab: dict
def calc_stack_history_batch_cpu(seq, stack_history, vocab):
    batch_size, length = seq.shape
    max_length = stack_history.shape[1]
    stack = np.full([batch_size, max_length], -1, np.int)
    stack_pointer = np.full([batch_size], 0, np.int)
    for i in range(length):
        stack_push_batch_cpu(
            seq=seq[:, i],
            stack=stack,
            pointer=stack_pointer,
            vocab=vocab,
        )
        stack_history[:, i, :] = stack
        stack_pop_batch_cpu(
            seq=seq[:, i],
            stack=stack,
            pointer=stack_pointer,
            vocab=vocab,
        )


def calc_nearest_batch_cpu(stack, pointer):
    tmp_pointer = np.maximum(0, pointer - 1)
    return stack[range(stack.shape[0]), tmp_pointer]


def check_in_stack_history(stack_history_k, tag, batch_idx, idx):
    for i in range(stack_history_k.shape[-1]):
        if stack_history_k[batch_idx][idx][i] == -1:
            break
        if tag == stack_history_k[batch_idx][idx][i]:
            return True
    return False


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# nearest_q: [batch, max_length]
# vocab: dict
def update_tgt_stack_batch_cpu(step, seq, stack,
                               stack_pointer, nearest_q,
                               stack_history_k=None, vocab=None):
    # update stack and history
    stack_push_batch_cpu(seq, stack, stack_pointer, vocab)
    nearest_q[:, step] = calc_nearest_batch_cpu(stack, stack_pointer)
    if stack_history_k is not None:
        stack_history_k[:, step, :] = stack
    stack_pop_batch_cpu(seq, stack, stack_pointer, vocab)


# mat: [batch, max_length, max_length]
# nearest_q: [batch, max_length]
# stack_history: [batch, max_length, max_length]
def update_dec_self_attn_batch_cpu(step, mat,
                                   nearest_q, stack_history_k):
    batch_size = mat.shape[0]
    # update mat
    for i in range(batch_size):
        for j in range(step + 1):
            # the down part
            if nearest_q[i][step] == -1:
                mat[i][step][j] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][step], i, j):
                mat[i][step][j] = 1
            else:
                mat[i][step][j] = 2
            # the right part
            if nearest_q[i][j] == -1:
                mat[i][j][step] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][j], i, step):
                mat[i][j][step] = 1
            else:
                mat[i][j][step] = 2


# mat: [batch, max_length, max_length]
# nearest_q: [batch, max_length]
# stack_history: [batch, max_length, max_length]
def update_enc_dec_attn_batch_cpu(step, length_k, mat,
                                  nearest_q, stack_history_k):
    batch_size = mat.shape[0]
    # update mat
    for i in range(batch_size):
        for j in range(length_k):
            if nearest_q[i][step] == -1:
                mat[i][step][j] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][step], i, j):
                mat[i][step][j] = 1
            else:
                mat[i][step][j] = 2


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# vocab: dict
def stack_push_batch_gpu(seq, stack, pointer, vocab):
    batch_size = seq.shape[0]
    for i in range(batch_size):
        if vocab["tag_type"][seq[i]] == -1:
            stack[i][pointer[i]] = vocab["tag_content"][seq[i]]
            pointer[i] += 1


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# vocab: dict
def stack_pop_batch_gpu(seq, stack, pointer, vocab):
    batch_size = seq.shape[0]
    for i in range(batch_size):
        if vocab["tag_type"][seq[i]] == 1 and pointer[i] > 0:
            stack[i][pointer[i] - 1] = -1 # for stack_history
            pointer[i] -= 1


# calculate batches of stack history (used for src)
# seq: [batch, length]
# stack_history: [batch, max_length, max_length]
# vocab: dict
def calc_stack_history_batch_gpu(seq, stack_history, vocab):
    batch_size, length = seq.shape
    max_length = stack_history.shape[1]
    stack = torch.full([batch_size, max_length], -1, dtype=torch.int)
    stack_pointer = torch.full([batch_size], 0, dtype=torch.int)
    for i in range(length):
        stack_push_batch_gpu(
            seq=seq[:, i],
            stack=stack,
            pointer=stack_pointer,
            vocab=vocab,
        )
        stack_history[:, i, :] = stack
        stack_pop_batch_gpu(
            seq=seq[:, i],
            stack=stack,
            pointer=stack_pointer,
            vocab=vocab,
        )


def calc_nearest_batch_gpu(stack, pointer):
    tmp_pointer = torch.clamp(pointer - 1, min=0)
    return torch.squeeze(stack.gather(1, tmp_pointer.view(-1, 1)))


# def check_in_stack_history(stack_history_k, tag, batch_idx, idx):
#     for i in range(stack_history_k.shape[-1]):
#         if stack_history_k[batch_idx][idx][i] == -1:
#             break
#         if tag == stack_history_k[batch_idx][idx][i]:
#             return True
#     return False


# seq: [batch]
# stack: [batch, max_length]
# pointer: [batch]
# nearest_q: [batch, max_length]
# vocab: dict
def update_tgt_stack_batch_gpu(step, seq, stack,
                               stack_pointer, nearest_q,
                               stack_history_k=None, vocab=None):
    # update stack and history
    stack_push_batch_gpu(seq, stack, stack_pointer, vocab)
    nearest_q[:, step] = calc_nearest_batch_gpu(stack, stack_pointer)
    if stack_history_k is not None:
        stack_history_k[:, step, :] = stack
    stack_pop_batch_gpu(seq, stack, stack_pointer, vocab)


# mat: [batch, max_length, max_length]
# nearest_q: [batch, max_length]
# stack_history: [batch, max_length, max_length]
def update_dec_self_attn_batch_gpu(step, mat,
                                   nearest_q, stack_history_k):
    batch_size = mat.shape[0]
    # update mat
    for i in range(batch_size):
        for j in range(step + 1):
            # the down part
            if nearest_q[i][step] == -1:
                mat[i][step][j] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][step], i, j):
                mat[i][step][j] = 1
            else:
                mat[i][step][j] = 2
            # the right part
            if nearest_q[i][j] == -1:
                mat[i][j][step] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][j], i, step):
                mat[i][j][step] = 1
            else:
                mat[i][j][step] = 2


# mat: [batch, max_length, max_length]
# nearest_q: [batch, max_length]
# stack_history: [batch, max_length, max_length]
def update_enc_dec_attn_batch_gpu(step, length_k, mat,
                                  nearest_q, stack_history_k):
    batch_size = mat.shape[0]
    # update mat
    for i in range(batch_size):
        for j in range(length_k):
            if nearest_q[i][step] == -1:
                mat[i][step][j] = 0
            elif check_in_stack_history(stack_history_k,
                                        nearest_q[i][step], i, j):
                mat[i][step][j] = 1
            else:
                mat[i][step][j] = 2


# helper print function
def print_state(step, src_seq, tgt_seq, src_vocab, tgt_vocab, state):
    print("batch_size=%d" % src_seq.shape[0])
    for i in range(src_seq.shape[0]):
        print("sentence %d" % i)
        print("src: ")
        print_sentence(src_seq[i], src_vocab["idx2word"])
        print("tgt: ")
        print_sentence(tgt_seq[i], tgt_vocab["idx2word"])
        print("stack_q: ")
        print(state["typed_matrix"]["stack_q"][i, :max(state["typed_matrix"]["stack_pointer_q"])])
        print("nearest_q: ")
        print(state["typed_matrix"]["nearest_q"][i, :step + 1])
        print("dec_self_attn mat: ")
        print(state["typed_matrix"]["dec_self_attn"]["mat"][i, :step + 1, :step + 1])
        print("enc_dec_attn mat: ")
        print(state["typed_matrix"]["enc_dec_attn"]["mat"][i, :step + 1, :src_seq.shape[1]])
        print()
    print()


def check_beam_search(src_seq, src_vocab, tgt_seq, tgt_vocab,
                      dec_self_attn, enc_dec_attn):
    dec_self_attn_0 = gen_typed_matrix_cpu(
        tgt_seq, tgt_seq, tgt_vocab, tgt_vocab).cpu().numpy()
    enc_dec_attn_0 = gen_typed_matrix_cpu(
        tgt_seq, src_seq, tgt_vocab, src_vocab).cpu().numpy()

    def merge_attn_mat(tensor):
        return np.squeeze(tensor[0]) * 0 + np.squeeze(tensor[1]) * 1 + np.squeeze(tensor[2]) * 2

    dec_self_attn_0 = merge_attn_mat(dec_self_attn_0)
    enc_dec_attn_0 = merge_attn_mat(enc_dec_attn_0)

    if not (dec_self_attn_0 == dec_self_attn).all():
        print(dec_self_attn_0)
        print(dec_self_attn)
        exit(1)

    if not (enc_dec_attn_0 == enc_dec_attn).all():
        print(enc_dec_attn_0)
        print(enc_dec_attn)
        exit(1)
