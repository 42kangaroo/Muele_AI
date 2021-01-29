import multiprocessing as mp
import unittest

import numpy as np

import MillEnv
import encoders
import mcts


class NetworkTest(unittest.TestCase):

    def setUp(self) -> None:
        import Network
        self.env = MillEnv.MillEnv()
        self.net = Network.get_net(96, 3, 256, 4, 1, 24, (8, 3, 4))

    def test_create(self):
        self.assertEqual(2, len(
            self.net(encoders.prepareForNetwork([self.env.board], [self.env.isPlaying], [self.env.moveNeeded],
                                                [self.env.gamePhase[1 if self.env.isPlaying == 1 else 0]],
                                                [self.env.selected]))))

    def test_fit(self):
        self.net.fit(encoders.prepareForNetwork([self.env.board], [self.env.isPlaying], [self.env.moveNeeded],
                                                [self.env.gamePhase[1 if self.env.isPlaying == 1 else 0]],
                                                [self.env.selected]),
                     {'policy_output': np.array(
                         [[.25, 0., 0.25, 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., .0, 0.,
                           -1., 0.]]),
                         'value_output': np.array([[0.5]])}, epochs=1,
                     batch_size=1)

    def test_loss(self):
        import Network
        print(Network.cross_entropy_with_logits(
            [.25, 0., 0.25, 0., 0., 0., 0., -1., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., .0, 0.,
             -1., 0.], [2., -1., 0.2, -1., -1., 0., 0., 0., 0., 0., 4., 0., 0., 1., 0., 0., 6., 0., 0., 1., .0, 0.,
                        5., 0.]))


class EncodersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MillEnv.MillEnv()
        self.random_gen = np.random.default_rng(seed=0)

    def test_shape(self):
        self.assertEqual((1, 8, 3, 4),
                         encoders.prepareForNetwork([self.env.board], [self.env.isPlaying], [self.env.moveNeeded],
                                                    [self.env.gamePhase[1 if self.env.isPlaying == 1 else 0]],
                                                    [self.env.selected]).shape)

    def test_reshape(self):
        self.env.makeMove(0)
        self.assertEqual(1, encoders.prepareForNetwork([self.env.board], [self.env.isPlaying], [self.env.moveNeeded],
                                                       [self.env.gamePhase[1 if self.env.isPlaying == 1 else 0]],
                                                       [self.env.selected])[0, 0, 0, 1])

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
        self.assertEqual(encoders.prepareForNetwork([self.env.board], [self.env.isPlaying], [self.env.moveNeeded],
                                                    [self.env.gamePhase[1 if self.env.isPlaying == 1 else 0]],
                                                    [self.env.selected])[0, 4, 0, 0], 2)

    def test_Swap(self):
        self.assertEqual(1, encoders.swapvalues([0, 1, 2], 1, 2)[2])

    def test_target_symetries(self):
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 0)[1][0], 21)
        self.assertEqual(encoders.getTargetSymetries(np.arange(24), 2)[4][0], 3)


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        import Network
        self.env = MillEnv.MillEnv()
        self.state = mcts.State(np.zeros((1, 24)), 0, -self.env.isPlaying, self.env)
        self.nnet = Network.get_net(24, 3, 64, 4, 1, 24, (8, 3, 4))

    def test_dirichlet(self):
        self.env.makeMove(1)
        self.state = mcts.State(np.zeros((1, 24)), 0, -self.env.isPlaying, self.env)
        self.state.add_noise(np.random.default_rng())
        self.assertNotEqual(self.state.priors[0, 0], 0)

    def test_expand(self):
        self.state, _ = self.state.expand(0, self.nnet)
        self.assertEqual(self.state.state[1], -1)
        self.assertEqual(len(self.state.valid_moves), 23)

    def test_best_move(self):
        self.state.add_noise(np.random.default_rng())
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
        import Network
        self.nnet = Network.get_net(96, 3, 256, 4, 1, 24, (8, 3, 4))
        self.mcts = mcts.MonteCarloTreeSearch(mcts.State(np.zeros((1, 24)), 0, -self.env.isPlaying, self.env))

    @unittest.skip("ray is used")
    def test_search(self):
        val = self.mcts.root.setValAndPriors(self.nnet)
        self.mcts.root.backpropagate(val)
        self.mcts.search(self.nnet)

    @unittest.skip("ray is used")
    def test_generate_play(self):
        memory = np.zeros((1000, 7), dtype=object)
        val = mp.Value("L")
        self.mcts.generatePlay(memory, self.nnet, val, 1, 1)

    @unittest.skip("ray is used")
    def test_fit_generated(self):
        from tensorflow import keras
        memory = np.zeros((48000, 7), dtype=object)
        val = mp.Value("L")
        self.mcts.generatePlay(memory, self.nnet, val, 1, 1)
        policy_out, board_in = np.zeros((val.value, 24), dtype=np.float32), np.zeros((val.value, 8, 3),
                                                                                     dtype=np.float32)
        for idx in range(val.value):
            policy_out[idx], board_in[idx] = memory[idx, 5], memory[idx, 0]
        tensorboard_callback = keras.callbacks.TensorBoard("TensorBoard", update_freq=2, profile_batch=0)
        self.nnet.fit(
            encoders.prepareForNetwork(board_in, memory[:val.value, 1], memory[:val.value, 4],
                                       memory[:val.value, 3],
                                       memory[:val.value, 2]),
            {'policy_output': policy_out,
             'value_output': memory[:val.value, 6].astype(np.float32)}, epochs=5,
            batch_size=128, callbacks=[tensorboard_callback])
        self.nnet.save("models/test_save")

    @unittest.skip("ray is used")
    def test_pit(self):
        print(mcts.pit(self.nnet, self.nnet, -1, 8, 1.15))

    @unittest.skip("mulitpocessing doesn't work, infinite loop")
    def test_multiprocessing(self):
        file = "file_mem://alphaMemoryMulti"
        val = mp.Value("L")
        processes = [mp.Process(target=self.mcts.generatePlay, args=(file, self.nnet, val, 1, 1)) for i in range(8)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


class SaveTest(unittest.TestCase):
    def test_save(self):
        import pickle
        array = np.array([1, 2, 3])
        i = 1
        np.save("test.npy", array)
        with open("test.obj", "wb") as f:
            pickle.dump(i, f)

    def test_load(self):
        array = np.load("test.npy")
        print(array)


if __name__ == '__main__':
    unittest.main()
