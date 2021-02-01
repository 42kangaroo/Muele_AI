import time

import ray
from ray.util.multiprocessing import Pool

import MillEnv
import Network
import configs
import encoders
import logger

ray.shutdown()
ray.init()
env = MillEnv.MillEnv()


def predict(nnet, envi, logi):
    logi.log.remote("hello")
    return ray.get(nnet.predict.remote(encoders.prepareForNetwork([envi.board], [envi.isPlaying], [envi.moveNeeded],
                                                                  [envi.gamePhase[1 if envi.isPlaying == 1 else 0]],
                                                                  [envi.selected])))


@ray.remote
def increment(que):
    idx = que.get(block=True)
    que.put(idx + 1)


if __name__ == "__main__":
    try:
        logi = logger.Logger.remote("test.log")
        handle = Network.ServeModels.options(max_concurrency=4).remote("models/test_weights", configs.FILTERS,
                                                                       configs.KERNEL_SIZE, configs.HIDDEN_SIZE,
                                                                       configs.OUT_FILTERS,
                                                                       configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS,
                                                                       configs.INPUT_SIZE, 4)
        with Pool(8) as pool:
            firstTime = time.time()
            print(pool.starmap(predict, [(handle, env, logi) for i in range(100)]), time.time() - firstTime)
        handle.kill_actors.remote()
        handle2 = Network.ServeModels.options(max_concurrency=4).remote("models/test_weights", configs.FILTERS,
                                                                        configs.KERNEL_SIZE, configs.HIDDEN_SIZE,
                                                                        configs.OUT_FILTERS,
                                                                        configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS,
                                                                        configs.INPUT_SIZE, 4)
        with Pool(8) as pool:
            firstTime = time.time()
            print(pool.starmap(predict, [(handle2, env, logi) for i in range(100)]), time.time() - firstTime)

    except Exception as e:
        print(e)
    finally:
        ray.shutdown()
