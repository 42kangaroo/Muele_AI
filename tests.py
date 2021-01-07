import unittest

import numpy as np

import MillEnv
import Network
import encoders
import mcts


class NetworkTest(unittest.TestCase):

    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.net = Network.Residual_Model(256, 24)

    def test_create(self):
        self.assertEqual(2, len(
            self.net(encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                self.env.selected))))


class EncodersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.random_gen = np.random.default_rng(seed=0)

    def test_shape(self):
        self.assertEqual((1, 8, 3, 4),
                         encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                    self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                    self.env.selected).shape)

    def test_reshape(self):
        self.env.makeMove(0)
        self.assertEqual(1, encoders.prepareForNetwork(self.env.board, self.env.isPlaying, self.env.moveNeeded,
                                                       self.env.gamePhase[1 if self.env.isPlaying == 1 else 0],
                                                       self.env.selected)[0, 0, 0, 1])

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
                                                    self.env.selected)[0, 4, 0, 0], 2)

    def test_Swap(self):
        self.assertEqual(1, encoders.swapvalues([0, 1, 2], 1, 2)[2])

    def test_target_symetries(self):
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 0)[1][0], 21)
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 2)[4][0], 3)


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.state = mcts.State(np.zeros((1, 24)), 0, self.env.isPlaying, self.env)
        self.nnet = Network.Residual_Model(256, 24)

    def test_dirichlet(self):
        self.env.makeMove(1)
        self.state = mcts.State(np.zeros((1, 24)), 0, self.env.isPlaying, self.env)
        self.state.add_noise()
        self.assertNotEqual(self.state.priors[0, 0], 0)

    def test_expand(self):
        self.state, _ = self.state.expand(0, self.nnet)
        self.assertEqual(self.state.state[1], -1)
        self.assertEqual(len(self.state.valid_moves), 23)

    def test_best_move(self):
        self.state.add_noise()
        self.assertEqual(self.state.best_child().last_move, 0)

    def test_backpropaget(self):
        self.state, val = self.state.expand(0, self.nnet)
        self.state.backpropagate(val)
        self.assertNotEqual(self.state.parent.q, 0)

    def test_pi(self):
        for i in range(25):
            node, val = self.state.expand(i % 24, self.nnet)
            node.backpropagate(val)
        self.assertEqual(np.argmax(self.state.pi(0)), 0)
        self.assertEqual(np.argmax(self.state.pi(1)), 0)


class MCTSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.nnet = Network.Residual_Model(256, 24)
        self.mcts = mcts.MonteCarloTreeSearch(mcts.State(np.zeros((1, 24)), 0, self.env.isPlaying, self.env))

    def test_search(self):
        val = self.mcts.root.setValAndPriors(self.nnet)
        self.mcts.root.backpropagate(val)
        self.mcts.search(self.nnet)


if __name__ == '__main__':
    unittest.main()
