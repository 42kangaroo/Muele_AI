from os.path import isfile

import ray
from tensorflow import keras

import Network
import configs
import encoders
import logger
import mcts
import memory

if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    logger_handle = logger.Logger.remote(configs.LOGGER_PATH)
    current_mem_size = configs.MIN_MEMORY
    mem = memory.Memory.remote(current_mem_size, configs.MAX_MEMORY)
    episode = 0
    current_Network = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                      configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    if isfile("interrupt_array.npy") and isfile("interrupted_vars.obj"):
        episode = ray.get(mem.loadState.remote("interrupt_array.npy", "interrupted_vars.obj"))
        current_Network.load_weights(configs.NETWORK_PATH + str(episode) + ".h5")
        logger_handle.log.remote("=========== restarting training ==========")
    try:
        while episode <= configs.TRAINING_LOOPS:
            logger_handle.log.remote("============== starting episode " + str(episode) + " ===============")
            current_Network_path = configs.NETWORK_PATH + str(episode) + ".h5"
            logger_handle.log.remote("saving actual net to " + current_Network_path)
            current_Network.save_weights(current_Network_path)
            logger_handle.log.remote("============== starting selfplay ================")
            futures_playGeneration = [
                mcts.execute_generate_play.remote(mem, current_Network_path, logger_handle, configs.SIMS_FAKTOR,
                                                  configs.SIMS_EXPONENT) for play in range(configs.EPISODES)]
            ray.get(futures_playGeneration)
            logger_handle.log.remote("============== starting training ================")
            train_data = ray.get(mem.getTrainSamples.remote())
            tensorboard_callback = keras.callbacks.TensorBoard(configs.TENSORBOARD_PATH + str(episode), update_freq=10,
                                                               profile_batch=0)
            current_Network.compile(optimizer='adam',
                                    loss={'policy_output': Network.cross_entropy_with_logits, 'value_output': 'mse'},
                                    loss_weights=[0.5, 0.5],
                                    metrics=['accuracy'])
            current_Network.fit(
                encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]),
                {'policy_output': train_data[5],
                 'value_output': train_data[6]}, epochs=configs.EPOCHS,
                batch_size=configs.BATCH_SIZE, callbacks=[tensorboard_callback])
            new_net_path = "temp_new_net.h5"
            current_Network.save_weights(new_net_path)
            logger_handle.log.remote("============ starting pit =============")
            futures_pit = [
                mcts.execute_pit.remote(current_Network_path, new_net_path, 1 if pit_iter % 2 == 0 else -1,
                                        logger_handle,
                                        configs.SIMS_FAKTOR,
                                        configs.SIMS_EXPONENT) for pit_iter in range(configs.EVAL_EPISODES)]
            if sum(ray.get(futures_pit)) < configs.SCORING_THRESHOLD:  # not better then previus
                logger_handle.log.remote("falling back to old network")
                current_Network.load_weights(current_Network_path)
            else:
                logger_handle.log.remote("new network accepted")
            if episode <= configs.MEMORY_ITERS:
                current_mem_size = int(
                    configs.MIN_MEMORY + (configs.MAX_MEMORY - configs.MIN_MEMORY) * (episode / configs.MEMORY_ITERS))
                logger_handle.log.remote("changed mem size to " + str(current_mem_size))
                mem.changeMaxSize.remote(current_mem_size)
            episode += 1
        logger_handle.log.remote("============ finisched AlphaZero ===========")
        current_Network.save_weights("best_net.h5")
        ray.get(mem.saveState.remote(episode, "finished_array.npy", "finisched_vars.obj"))
        ray.shutdown()
    except KeyboardInterrupt:
        logger_handle.log.remote("============ interupted training ===========")
        ray.get(mem.saveState.remote(episode, "interrupt_array.npy", "interrupted_vars.obj"))
        ray.shutdown()
