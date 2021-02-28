import gc
import multiprocessing
import os

import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras.backend import clear_session

import MillEnv
import configs
import encoders
import mcts
import memory


def predict(envi, nnet_path):
    import Network
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


def test_predict_multi(env, path):
    import Network
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(path)
    print("Loaded")
    x = nnet(encoders.prepareForNetwork([env.board], [env.isPlaying], [env.moveNeeded],
                                        [env.gamePhase[1 if env.isPlaying == 1 else 0]],
                                        [env.selected]))
    print("predicted")
    return x


@ray.remote(num_cpus=1, num_gpus=0)
def test_x():
    import Network
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


if __name__ == "__man__":
    import Network
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

if __name__ == "__man__":
    import Network

    mem = memory.Memory(1000, 1000)
    _ = mem.loadState("interrupt_array.npy", "interrupted_vars.obj")
    current_Network = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE,
                                      configs.OUT_FILTERS,
                                      configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    current_Network.load_weights(configs.BEST_PATH)
    current_Network.compile(optimizer='adam',
                            loss={'policy_output': Network.cross_entropy_with_logits, 'value_output': 'mse'},
                            loss_weights=[0.5, 0.5],
                            metrics=['accuracy'])

    for i in range(100):
        train_data = mem.getTrainSamples()
        current_Network.fit(
            encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3],
                                       train_data[4]),
            {'policy_output': train_data[5],
             'value_output': train_data[6]}, epochs=configs.EPOCHS,
            batch_size=configs.BATCH_SIZE)
        current_Network.save_weights(configs.NEW_NET_PATH)
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

if __name__ == "__man__":
    import ray_funcs

    for i in range(100):
        ray.init(num_cpus=1)
        futures = [ray_funcs.printer.remote()]
        ray.get(futures)
        ray.shutdown()

if __name__ == "__main__":
    env = MillEnv.MillEnv()
    multiprocessing.Process(target=test_predict_multi, args=(env, "run3/models/best_net.h5"))
    with multiprocessing.Pool(8) as pool:
        pool.starmap(predict, [(env, "run3/models/best_net.h5") for i in range(16)])
