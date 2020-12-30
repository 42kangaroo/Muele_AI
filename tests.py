import unittest

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

    def test_shape(self):
        self.assertEqual((8, 3, 4), encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                               self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                               self.env.selected).shape)

    def test_reshape(self):
        self.env.makeMove(0)
        self.assertEqual(1, encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                       self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                       self.env.selected)[0, 0, 1])


if __name__ == '__main__':
    unittest.main()
