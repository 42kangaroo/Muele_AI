import numpy as np


def shortenBoard(board):
    shortend = ((board + 1) * np.logspace(0, 23, base=3, num=24)).sum()
    return shortend


def lengthenBoard(shortend):
    board = (shortend % np.logspace(1, 24, base=3, num=24) // np.logspace(0, 23, base=3, num=24)) - 1
    return board


def prepareForNetwork(board, player, moveNeeded, gamePhase, selected=None):
    networkState = np.moveaxis(np.array(
        [board == 1 * player, board == -1 * player, np.full(24, gamePhase), np.full(24, moveNeeded)],
        dtype=np.float32).reshape((4, 8, 3)), 0, -1)
    if gamePhase > 0 and selected:
        networkState[selected] = 2
    return networkState
