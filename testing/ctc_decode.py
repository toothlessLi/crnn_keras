import numpy as np


def ctc_decode(prob, charset):
    N, T, C = prob.shape
    # in this demo, N always = 1 at test
    blank_idx = C - 1
    out = ''
    for i in range(T):
        # blank is last one in the charset
        ind = blank_idx
        last = ind
        s = prob[0, i, ind]
        if s > 0.5:
            continue
        for c in range(C-1):
            if prob[0, i, c] > s:
                ind = c
                s = prob[0, i, c]
        if ind != blank_idx and last != ind:
            out += charset[ind]
        last = ind
    return out
