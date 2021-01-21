import keras
import ray

import Network
import configs
import encoders
import mcts
import memory

ray.shutdown()
ray.init()

current_Network = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                  configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
current_mem_size = configs.MIN_MEMORY
mem = memory.Memory.remote(current_mem_size, configs.MAX_MEMORY)
for episode in range(configs.TRAINING_LOOPS):
    current_Network_path = configs.NETWORK_PATH + str(episode) + ".h5"
    current_Network.save_weights(current_Network_path)
    futures_playGeneration = [
        mcts.execute_generate_play.remote(mem, current_Network_path, configs.SIMS_FAKTOR,
                                          configs.SIMS_EXPONENT) for play in range(configs.EPISODES)]
    ray.get(futures_playGeneration)
    train_data = ray.get(mem.getTrainSamples.remote())
    tensorboard_callback = keras.callbacks.TensorBoard("TensorBoard/episode" + str(episode), update_freq=10,
                                                       profile_batch=0)
    current_Network.fit(
        encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]),
        {'policy_output': train_data[5],
         'value_output': train_data[6]}, epochs=configs.EPOCHS,
        batch_size=configs.BATCH_SIZE)  # , callbacks=[tensorboard_callback])
    new_net_path = "models/temp_new_net.h5"
    current_Network.save_weights(new_net_path)
    futures_pit = [
        mcts.execute_pit.remote(current_Network_path, new_net_path, 1 if pit_iter % 2 == 0 else -1, configs.SIMS_FAKTOR,
                                configs.SIMS_EXPONENT) for pit_iter in range(configs.EVAL_EPISODES)]
    if sum(ray.get(futures_pit)) < configs.SCORING_THRESHOLD:  # not better then previus
        current_Network.load_weights(current_Network_path)
    if episode <= configs.MEMORY_ITERS:
        current_mem_size = int(
            configs.MIN_MEMORY + (configs.MAX_MEMORY - configs.MIN_MEMORY) * (episode / configs.MEMORY_ITERS))
        mem.changeMaxSize.remote(current_mem_size)
ray.shutdown()
