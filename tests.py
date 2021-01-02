import unittest

import numpy as np

import MillEnv
import Network
import encoders


class NetworkTest(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.net = Network.Residual_Model(128, 24)

    def test_create(self):
        self.assertEqual(2, len(
            self.net(encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                self.env.selected).reshape(1, 8, 3, 4))))


class EncodersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.random_gen = np.random.default_rng(seed=0)

    def test_shape(self):
        self.assertEqual((8, 3, 4), encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                               self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                               self.env.selected).shape)

    def test_reshape(self):
        self.env.makeMove(0)
        self.assertEqual(1, encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                       self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                       self.env.selected)[0, 0, 1])

    def test_symetries(self):
        self.env.makeMove(0)
        self.env.makeMove(2)
        symetrys = encoders.getSymetries(self.env.board, self.env.selected)
        self.assertEqual(symetrys[0][4, 0, 0], 1)

    def test_symetries_selected(self):
        while self.env.moveNeeded != 2:
            self.env.makeMove(self.random_gen.choice(self.env.getValidMoves()))
        symetrys = encoders.getSymetries(self.env.board, self.env.selected)
        self.assertEqual(symetrys[1][2], (4, 2))

    def test_selected_reschape(self):
        while self.env.moveNeeded != 2:
            self.env.makeMove(self.random_gen.choice(self.env.getValidMoves()))
        self.assertEqual(encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                    self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                    self.env.selected)[4, 0, 0], 2)

    def test_Swap(self):
        self.assertEqual(1, encoders.swapvalues([0, 1, 2], 1, 2)[2])

    def test_target_symetries(self):
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 0)[1][0], 21)
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 2)[4][0], 3)


if __name__ == '__main__':
    unittest.main()
