import ray

import MillEnv
import Network
import configs_fake as configs
import encoders
import logger

actual_nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                              configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
actual_nnet.summary()
ray.shutdown()
ray.init()

env = MillEnv.MillEnv()
actual_nnet.save_weights("models/test_weights")


@ray.remote
def predict(nnet_weights_path, envi, logi):
    import Network
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(nnet_weights_path)
    logi.log.remote("hello")
    return nnet(encoders.prepareForNetwork([envi.board], [envi.isPlaying], [envi.moveNeeded],
                                           [envi.gamePhase[1 if envi.isPlaying == 1 else 0]],
                                           [envi.selected]))


@ray.remote
def increment(que):
    idx = que.get(block=True)
    que.put(idx + 1)


if __name__ == "__main__":
    logi = logger.Logger.remote("test.log")
    futures = [predict.remote("models/test_weights", env, logi) for i in range(2)]
    ray.get(futures)
ray.shutdown()
