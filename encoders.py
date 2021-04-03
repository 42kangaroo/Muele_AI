import copy

import numpy as np


def shortenBoard(board):
    shortend = ((board + 1) * np.logspace(0, 23, base=3, num=24)).sum()
    return shortend


def lengthenBoard(shortend):
    board = (shortend % np.logspace(1, 24, base=3, num=24) // np.logspace(0, 23, base=3, num=24)) - 1
    return board


def prepareForNetwork(boards, player, moveNeeded, gamePhase, selected=None):
    states = np.array([[board == 1 * play, board == -1 * play, np.full((8, 3), gameP), np.full((8, 3), moveN)] for
                       board, play, gameP, moveN in
                       zip(np.array(boards).reshape(-1, 8, 3), player, gamePhase, moveNeeded)], dtype=np.float32)
    networkState = np.moveaxis(states
                               , 1, -1)
    for i, gameP, sel in zip(range(len(gamePhase)), gamePhase, selected):
        if gameP > 0 and sel:
            if not isinstance(sel, tuple):
                sel = sel // 3, sel % 3, 0
            networkState[i, sel] = 2
    return networkState.reshape(-1, 8, 3, 4)


def getSymetries(board: np.ndarray, selected=None):
    columns = (np.array([[0, 3, 7],
                         [1, 3, 6],
                         [2, 3, 5],
                         [0, 1, 2],
                         [5, 6, 7],
                         [2, 4, 5],
                         [1, 4, 6],
                         [0, 4, 7]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 2, 0],
                         [1, 1, 1],
                         [1, 1, 1],
                         [2, 0, 2],
                         [2, 1, 2],
                         [2, 2, 2]]))
    board = board.reshape(8, 3)
    symetrys = np.array([board, board[::-1], board[:, ::-1], board[::-1, ::-1], board[columns], board[columns][::-1],
                         board[columns][:, ::-1], board[columns][::-1, ::-1]])
    selectedArray = np.full(8, None)
    if selected:
        if not isinstance(selected, tuple):
            selected = selected // 3, selected % 3
        selectedTurned = (columns[0][selected], columns[1][selected])
        selectedArray = [selected, (-selected[0] + 7, selected[1]), (selected[0], -selected[1] + 2),
                         (-selected[0] + 7, -selected[1] + 2), selectedTurned,
                         (-selectedTurned[0] + 7, selectedTurned[1]),
                         (selectedTurned[0], -selectedTurned[1] + 2),
                         (-selectedTurned[0] + 7, -selectedTurned[1] + 2)]
    return symetrys, selectedArray


def getTargetSymetries(targets: np.ndarray, moveNeeded):
    if moveNeeded != 2:
        return getSymetries(targets)[0].reshape(-1, 24)
    targetsTurned = swapvalues(swapvalues(targets, 0, 3), 1, 2)
    return [targets, swapvalues(targets, 0, 2), swapvalues(targets, 1, 3), swapvalues(swapvalues(targets, 0, 2), 1, 3),
            targetsTurned, swapvalues(targetsTurned, 0, 2), swapvalues(targetsTurned, 1, 3),
            swapvalues(swapvalues(targetsTurned, 0, 2), 1, 3)]


def swapvalues(a, idx1, idx2):
    a = copy.deepcopy(a)
    a[idx1], a[idx2] = a[idx2], a[idx1]
    return a
