import time

import ray

import MillEnv
import Network
import configs
import encoders

actual_nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                              configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
actual_nnet.summary()
ray.init()

env = MillEnv.MillEnv()
actual_nnet.save_weights("models/test_weights")


@ray.remote
class Value(object):
    def __init__(self):
        self.value = 0

    def increment(self, val):
        vali = self.value
        time.sleep(0.001)
        self.value = val + vali

    def getVal(self):
        return self.value


@ray.remote
def predict(nnet_weights_path, envi, refi):
    import Network
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(nnet_weights_path)
    refi.increment.remote(1)
    return nnet(encoders.prepareForNetwork([envi.board], [envi.isPlaying], [envi.moveNeeded],
                                           [envi.gamePhase[1 if envi.isPlaying == 1 else 0]],
                                           [envi.selected]))


@ray.remote
def increment(refi):
    refi.increment.remote(1)


ref = Value.remote()
futi = [increment.remote(ref) for i in range(10000)]
fut = [predict.remote("models/test_weights", env, ref) for i in range(8)]
print(ray.get(fut))
ray.get(futi)
print(ray.get(ref.getVal.remote()))
