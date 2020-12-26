import numpy as np


def shortenBoard(board):
    shortend = ((board + 1) * np.logspace(0, 23, base=3, num=24)).sum()
    return shortend


def lengthenBoard(shortend):
    board = (shortend % np.logspace(1, 24, base=3, num=24) // np.logspace(0, 23, base=3, num=24)) - 1
    return board


def prepareForNetwork(board, player, moveNeeded, gamePhase, selected=None):
    networkState = np.array([board == 1 * player, board == -1 * player], dtype=int).reshape((2, -1, 3))
    networkState = np.append(networkState, np.full((1, 8, 3), gamePhase), axis=0)
    networkState = np.append(networkState, np.full((1, 8, 3), moveNeeded), axis=0)
    if gamePhase > 0 and selected:
        networkState[0, selected] = 2
    return networkState
