import ray
import ray

import MillEnv
import Network
import configs
import encoders
import mcts
import memory

actual_nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                              configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
actual_nnet.summary()
ray.shutdown()
ray.init()

env = MillEnv.MillEnv()
actual_nnet.save_weights("models/test_weights")


@ray.remote
def predict(nnet_weights_path, envi):
    import Network
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(nnet_weights_path)
    return nnet(encoders.prepareForNetwork([envi.board], [envi.isPlaying], [envi.moveNeeded],
                                           [envi.gamePhase[1 if envi.isPlaying == 1 else 0]],
                                           [envi.selected]))


@ray.remote
def increment(que):
    idx = que.get(block=True)
    que.put(idx + 1)


mem = memory.Memory.remote(4800, 6000)
futures = [mcts.execute_generate_play.remote(mem, "models/test_weights", 1, 1) for i in range(2)]
ray.get(futures)
train_data = ray.get(mem.getTrainSamples.remote())
tensorboard_callback = keras.callbacks.TensorBoard("TensorBoard", update_freq=2, profile_batch=0)
actual_nnet.fit(encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]),
                {'policy_output': train_data[5],
                 'value_output': train_data[6]}, epochs=5,
                batch_size=128, callbacks=[tensorboard_callback])
