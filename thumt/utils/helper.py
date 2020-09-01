import logging
import numpy as np


def print_sentence(sentence, idx2word):
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


def gen_typed_matrix(seq_q, seq_k, vocab_q, vocab_k):
    nq = len(seq_q)
    nk = len(seq_k)
    nearest_q = [None] * nq
    nearest_k = [None] * nk
    stack = []
    for i, x in enumerate(seq_q):
        if vocab_q["tagtype"][x] == "start":
            stack.append(vocab_q["tagcontent"][x])
        elif vocab_q["tagtype"][x] == "end":
            if len(stack) > 0:
                nearest_q[i] = stack[-1]
                stack.pop()
        if len(stack) > 0 and nearest_q[i] is None:
            nearest_q[i] = stack[-1]
        # print("vocab=%s tagtype=%s tagcontent=%s" % (vocab_q["idx2word"][x], vocab_q["tagtype"][x], vocab_q["tagcontent"][x]))
        # print(stack)
    stack.clear()
    for i, x in enumerate(seq_k):
        if vocab_k["tagtype"][x] == "start":
            stack.append(vocab_k["tagcontent"][x])
        elif vocab_k["tagtype"][x] == "end":
            if len(stack) > 0:
                nearest_k[i] = stack[-1]
                stack.pop()
        if len(stack) > 0 and nearest_k[i] is None:
            nearest_k[i] = stack[-1]
    # 0: std, 1: in, 2: out
    typed_matrix = np.zeros([3, nq, nk])
    for i in range(nq):
        for j in range(nk):
            if nearest_q[i] is None:
                typed_matrix[0][i][j] = 1
                print("0 ", end="")
            elif nearest_q[i] == nearest_k[j]:
                typed_matrix[1][i][j] = 1
                print("1 ", end="")
            else:
                typed_matrix[2][i][j] = 1
                print("2 ", end="")
        print()
    print(" ".join(["None" if s is None else s for s in nearest_q]))
    print_sentence(seq_q, vocab_q["idx2word"])
    print(" ".join(["None" if s is None else s for s in nearest_k]))
    print_sentence(seq_k, vocab_k["idx2word"])
    print()
    return typed_matrix
