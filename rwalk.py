from pylab import ceil, random

def init(states):
    position = int(ceil(states / 2) - 1)
    return position

def sample(S, A, states):
    if not A in (0, 1):
        print('Invalid action:', A)
        raise Exception
    S += 2 * A - 1

    if S == -1:
        R = -1.0
        S = None
    elif S == states:
        R = 1.0
        S = None
    else:
        R = 0.0

    return R, S