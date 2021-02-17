import copy
from typing import List

import numpy as np


class MillEnv(object):
    def __init__(self):
        self.isPlaying: int = 1
        self.gamePhase: list = [0, 0]
        self.moveNeeded: int = 0
        self.inHand: list = [9, 9]
        self.onBoard: list = [0, 0]
        self.checkerPositions: list = [[], []]
        self.selected = None
        self.board: np.ndarray = np.zeros(24)
        self.winner = 0
        self.columns: np.ndarray = np.array(
            [[0, 1, 2],  # for determining how many checkers from each player are in a row
             [3, 4, 5],
             [6, 7, 8],
             [9, 10, 11],
             [12, 13, 14],
             [15, 16, 17],
             [18, 19, 20],
             [21, 22, 23],
             [0, 9, 21],
             [3, 10, 18],
             [6, 11, 15],
             [1, 4, 7],
             [16, 19, 22],
             [8, 12, 17],
             [5, 13, 20],
             [2, 14, 23],
             ])
        self.previusStates = np.array([np.zeros(24)])

    def makeMove(self, move: int) -> (bool, int):
        if self.winner == 0:
            valid: bool = False
            last_state: tuple = self.getSummary(self.isPlaying)
            last_player: int = self.isPlaying
            reward: int = 0
            if self.moveNeeded == 0:  # Set Checker on position
                if self.board[move] == 0:
                    self.board[move] = self.isPlaying
                    self.checkerPositions[1 if self.isPlaying == 1 else 0].append(move)
                    valid = True
                    if self.gamePhase[1 if self.isPlaying == 1 else 0] == 2:
                        self.board[self.selected] = 0
                        self.moveNeeded = 1
                        self.checkerPositions[1 if self.isPlaying == 1 else 0].remove(self.selected)
                        if (self.board == self.previusStates).all(axis=1).any():
                            self.winner = 2
                            self.gamePhase = [3, 3]
                    else:
                        self.inHand[1 if self.isPlaying == 1 else 0] -= 1
                        self.onBoard[1 if self.isPlaying == 1 else 0] += 1
                        if np.array(self.inHand).sum() == 0:
                            self.gamePhase = [1, 1]
                            self.moveNeeded = 1
                    self.isPlaying = -self.isPlaying
            elif self.moveNeeded == 1:  # choose checker to move
                if move in self.getValidMoves():
                    self.previusStates = np.append(self.previusStates, [copy.deepcopy(self.board)], axis=0)
                    valid = True
                    self.selected = move
                    if self.gamePhase[1 if self.isPlaying == 1 else 0] == 2:
                        self.moveNeeded = 0
                    else:
                        self.moveNeeded = 2
            elif self.moveNeeded == 2:  # move checker up, down, left or right
                if self.getMoveFields(self.selected)[move] == 0:
                    idxToMoveAxis: np.ndarray = np.where(self.getInRows(self.selected) == self.selected)
                    idxToMove: list = list(zip(idxToMoveAxis[1], idxToMoveAxis[0]))
                    order = idxToMove[0][1]
                    valid = True
                    self.board[self.selected] = 0
                    last_state = self.getSummary(last_player)
                    self.checkerPositions[1 if self.isPlaying == 1 else 0].remove(self.selected)
                    if move == 0:  # up
                        self.board[self.getInRows(self.selected)[1][idxToMove[abs(order - 1)][0] - 1]] = self.isPlaying
                        self.checkerPositions[1 if self.isPlaying == 1 else 0].append(
                            self.getInRows(self.selected)[1][idxToMove[abs(order - 1)][0] - 1])
                    if move == 1:  # right
                        self.board[self.getInRows(self.selected)[0][idxToMove[order][0] + 1]] = self.isPlaying
                        self.checkerPositions[1 if self.isPlaying == 1 else 0].append(
                            self.getInRows(self.selected)[0][idxToMove[order][0] + 1])
                    if move == 2:  # down
                        self.board[self.getInRows(self.selected)[1][idxToMove[abs(order - 1)][0] + 1]] = self.isPlaying
                        self.checkerPositions[1 if self.isPlaying == 1 else 0].append(
                            self.getInRows(self.selected)[1][idxToMove[abs(order - 1)][0] + 1])
                    if move == 3:  # left
                        self.board[self.getInRows(self.selected)[0][idxToMove[order][0] - 1]] = self.isPlaying
                        self.checkerPositions[1 if self.isPlaying == 1 else 0].append(
                            self.getInRows(self.selected)[0][idxToMove[order][0] - 1])
                    self.selected = None
                    self.moveNeeded = 1
                    self.isPlaying = -self.isPlaying
                    if (self.board == self.previusStates).all(axis=1).any():
                        self.winner = 2
                        self.gamePhase = [3, 3]
            elif self.moveNeeded == 3:  # delete opponent checker
                threeInChosenRow: np.ndarray = abs(self.board[self.getInRows(move)].sum(axis=1)) == 3
                if self.board[move] == -1 * self.isPlaying and ~threeInChosenRow.any():
                    reward = 1
                    valid = True
                    self.board[move] = 0
                    self.checkerPositions[1 if self.isPlaying == -1 else 0].remove(move)
                    self.onBoard[1 if self.isPlaying == -1 else 0] -= 1
                    self.previusStates = np.array([np.zeros(24)])
                    if self.gamePhase[1 if self.isPlaying == -1 else 0] == 0:
                        self.moveNeeded = 0
                    elif self.gamePhase[1 if self.isPlaying == -1 else 0] == 1:
                        if self.onBoard[1 if self.isPlaying == -1 else 0] == 3:
                            self.gamePhase[1 if self.isPlaying == -1 else 0] = 2
                        self.moveNeeded = 1
                    elif self.gamePhase[1 if self.isPlaying == -1 else 0] == 2:
                        self.gamePhase = [3, 3]
                        self.winner = last_player
                    self.isPlaying = -self.isPlaying
            if last_state[0] < self.getSummary(last_player)[0]:
                self.isPlaying = last_player
                self.moveNeeded = 3
                canDelete: bool = False
                for pos in self.checkerPositions[1 if self.isPlaying == -1 else 0]:
                    if ~(abs(self.board[self.getInRows(pos)].sum(axis=1)) == 3).any():
                        canDelete = True
                        break
                if not canDelete:
                    valid = True
                    self.isPlaying = -self.isPlaying
                    if self.gamePhase[1 if self.isPlaying == -1 else 0] == 0:
                        self.moveNeeded = 0
                    elif self.gamePhase[1 if self.isPlaying == -1 else 0] >= 1:
                        self.moveNeeded = 1
            if self.gamePhase[1 if last_player == -1 else 0] == 1:
                finished = True
                for pos in self.checkerPositions[1 if last_player == -1 else 0]:
                    if ~(self.getMoveFields(pos).all()):
                        finished = False
                        break
                if finished:
                    self.winner = last_player
                    self.gamePhase = [3, 3]
            return valid, reward
        return False, 0

    def isFinished(self):
        return self.winner

    def getBoard(self) -> np.ndarray:
        return self.board

    def getInRows(self, pos: int) -> np.ndarray:
        arrayPos: np.ndarray = self.columns == pos
        return self.columns[arrayPos.any(axis=1)]

    def reset(self):
        self.isPlaying: int = 1
        self.gamePhase: list = [0, 0]
        self.moveNeeded: int = 0
        self.inHand: list = [9, 9]
        self.onBoard: list = [0, 0]
        self.checkerPositions: list = [[], []]
        self.selected = None
        self.board: np.ndarray = np.zeros(24)
        self.winner = 0

    def getSummary(self, player: int) -> (int, int):
        columnSums: np.ndarray = self.board[self.columns].sum(axis=1)
        numThreePlayerOpponent = np.count_nonzero(columnSums == -3 * player)
        numThreePlayerActual = np.count_nonzero(columnSums == 3 * player)
        return numThreePlayerActual, numThreePlayerOpponent

    def getMoveFields(self, pos: int) -> np.ndarray:
        moveFields: np.ndarray = np.zeros(4)
        chosenRows: np.ndarray = self.getInRows(pos)
        idxAxis: np.ndarray = np.where(chosenRows == pos)
        idx = list(zip(idxAxis[1], idxAxis[0]))
        order = idx[0][1]
        if idx[order][0] != 1:
            if idx[order][0] == 0:
                moveFields[3] = 2
                moveFields[1] = self.board[chosenRows[0][1]]
            elif idx[order][0] == 2:
                moveFields[1] = 2
                moveFields[3] = self.board[chosenRows[0][1]]
        else:
            moveFields[3] = self.board[chosenRows[0][0]]
            moveFields[1] = self.board[chosenRows[0][2]]
        if idx[abs(order - 1)][0] != 1:
            if idx[abs(order - 1)][0] == 0:
                moveFields[0] = 2
                moveFields[2] = self.board[chosenRows[1][1]]
            elif idx[abs(order - 1)][0] == 2:
                moveFields[2] = 2
                moveFields[0] = self.board[chosenRows[1][1]]
        else:
            moveFields[0] = self.board[chosenRows[1][0]]
            moveFields[2] = self.board[chosenRows[1][2]]
        return moveFields

    def getValidMoves(self) -> List[int]:
        validList = []
        if self.moveNeeded == 0:
            validList = np.arange(24)
            validList = validList[self.board.flat == 0]
        elif self.moveNeeded == 1:
            validList = [pos for pos in self.checkerPositions[1 if self.isPlaying == 1 else 0] if
                         ~self.getMoveFields(pos).all() or self.gamePhase[1 if self.isPlaying == 1 else 0] == 2]
        elif self.moveNeeded == 2:
            moveFields = self.getMoveFields(self.selected)
            validList = [move for move in np.arange(4) if moveFields[move] == 0]
        elif self.moveNeeded == 3:
            validList = [pos for pos in self.checkerPositions[1 if self.isPlaying == -1 else 0] if
                         ~(abs(self.board[self.getInRows(pos)].sum(axis=1)) == 3).any()]
        return validList

    def getFullState(self):
        return self.getBoard(), self.isPlaying, self.gamePhase, self.moveNeeded, self.inHand, self.onBoard, self.checkerPositions, self.selected, self.winner, self.previusStates

    def setFullState(self, board, isPlaying, gamePhase, moveNeded, inHand, onboard, checkerPositions, selected, winner,
                     previusStates):
        self.isPlaying: int = isPlaying
        self.gamePhase: list = copy.deepcopy(gamePhase)
        self.moveNeeded: int = moveNeded
        self.inHand: list = copy.deepcopy(inHand)
        self.onBoard: list = copy.deepcopy(onboard)
        self.checkerPositions: list = copy.deepcopy(checkerPositions)
        self.selected: int = selected
        self.board: np.ndarray = copy.deepcopy(board)
        self.winner: int = winner
        self.previusStates = copy.deepcopy(previusStates)
