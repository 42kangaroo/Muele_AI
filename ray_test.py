import ray

import MillEnv
import Network
import configs
import encoders
import numpy as np
import os
import mcts
import gc
from tensorflow.keras.backend import clear_session
import gc
import os

import numpy as np
import ray
from tensorflow.keras.backend import clear_session

import MillEnv
import Network
import configs
import encoders
import mcts


@ray.remote(num_cpus=1, num_gpus=0)
def predict(envi, nnet_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(nnet_path)
    z = mcts.MonteCarloTreeSearch(mcts.State(np.zeros((1, 24)), 1, 1, envi))
    z.search(nnet, 4, 1)
    print(z.root.q)
    del nnet
    del z
    gc.collect()
    clear_session()
    gc.collect()
    return np.zeros((1000, 100, 100))


@ray.remote(num_cpus=1, num_gpus=0)
def test_x():
    envi = MillEnv.MillEnv()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    x = mcts.State(np.zeros((1, 24)), 1, 1, envi)
    for i in range(24):
        x.expand(i, nnet)
    return True


class test:
    def call(self, arg):
        self.x = np.zeros((1000, 100, 100))
        self.y = arg(encoders.prepareForNetwork([np.zeros(24)], [1], [0],
                                                [0], [None]))
        return np.zeros((1000, 100, 100))


if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    env = MillEnv.MillEnv()
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    try:
        nnet.save_weights("models/gnegnegne.h5")
        futures_playGeneration = [predict.remote(env, "models/gnegnegne.h5") for i in range(1000)]
        finished, not_finished = ray.wait(futures_playGeneration)
        i = 0
        while not_finished:
            print("hello World", i)
            i += 1
            futures_playGeneration = not_finished
            finished, not_finished = ray.wait(futures_playGeneration)
    except BaseException as e:
        print("catched")
        print(e.__doc__)
    finally:
        ray.shutdown()
