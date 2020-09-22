import logging
import numpy as np
import torch


def print_sentence(sentence, idx2word):
    if isinstance(sentence, torch.Tensor):
        sentence = sentence.cpu().numpy()
    if sentence.ndim == 1:
        tokens = [idx2word[id] for id in sentence]
        print(" ".join(tokens))
    elif sentence.ndim == 2:
        if len(sentence) > 5:
            sentence = sentence[:5, :]
        for i in range(len(sentence)):
            tokens = [idx2word[id] for id in sentence[i]]
            print(" ".join(tokens))
    else:
        raise ValueError("Can't handle more than 2 dimensions")


def gen_typed_matrix_cpu(seq_q, seq_k, vocab_q, vocab_k):
    batch, nq = seq_q.shape
    _, nk = seq_k.shape
    nearest_q = [["" for _ in range(nq)] for _ in range(batch)]
    stack = [[] for _ in range(batch)]

    for i in range(batch):
        for j in range(nq):
            x = seq_q[i][j]
            if vocab_q["tag_type_str"][x] == "start":
                stack[i].append(vocab_q["tag_content_str"][x])
            if len(stack[i]) == 0:
                nearest_q[i][j] = ""
            else:
                nearest_q[i][j] = stack[i][-1]
            if vocab_q["tag_type_str"][x] == "end" and len(stack[i]) > 0:
                stack[i].pop()

    typed_matrix = np.zeros([3, batch, nq, nk])
    stack = [[] for _ in range(batch)]
    for i in range(batch):
        for j in range(nk):
            x = seq_k[i][j]
            if vocab_k["tag_type_str"][x] == "start":
                stack[i].append(vocab_k["tag_content_str"][x])
            for k in range(nq):
                # 0: std, 1: in, 2: out
                if nearest_q[i][k] == "":
                    typed_matrix[0][i][k][j] = 1
                elif nearest_q[i][k] in stack:
                    typed_matrix[1][i][k][j] = 1
                else:
                    typed_matrix[2][i][k][j] = 1
            if vocab_k["tag_type_str"][x] == "end" and len(stack[i]) > 0:
                stack[i].pop()

    return torch.from_numpy(typed_matrix).float().cuda()


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
