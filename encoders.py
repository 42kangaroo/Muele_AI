import numpy as np


def shortenBoard(board):
    shortend = ((board + 1) * np.logspace(0, 23, base=3, num=24)).sum()
    return shortend


def lengthenBoard(shortend):
    board = (shortend % np.logspace(1, 24, base=3, num=24) // np.logspace(0, 23, base=3, num=24)) - 1
    return board
