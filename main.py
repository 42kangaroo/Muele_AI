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
    logger_handle = logger.Logger(configs.LOGGER_PATH)
    current_mem_size = configs.MIN_MEMORY
    mem = memory.Memory(current_mem_size, configs.MAX_MEMORY)
    episode = 0
    current_Network = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                      configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    if isfile("interrupt_array.npy") and isfile("interrupted_vars.obj"):
        episode = mem.loadState("interrupt_array.npy", "interrupted_vars.obj")
        current_Network.load_weights(configs.NETWORK_PATH + str(episode) + ".h5")
        logger_handle.log("=========== restarting training ==========")
    try:
        while episode <= configs.TRAINING_LOOPS:
            logger_handle.log("============== starting episode " + str(episode) + " ===============")
            current_Network_path = configs.NETWORK_PATH + str(episode) + ".h5"
            logger_handle.log("saving actual net to " + current_Network_path)
            current_Network.save_weights(current_Network_path)
            logger_handle.log("============== starting selfplay ================")
            futures_playGeneration = [
                mcts.execute_generate_play.remote(current_Network_path, configs.SIMS_FAKTOR,
                                                  configs.SIMS_EXPONENT) for play in range(configs.EPISODES)]
            finished, not_finished = ray.wait(futures_playGeneration)
            while not_finished:
                stmem = ray.get(finished)[0]
                mem.addToMem(stmem)
                logger_handle.log("player won: " + str(-stmem[0][6]) + " turns played: " + str(len(stmem) // 8))
                futures_playGeneration = not_finished
                finished, not_finished = ray.wait(futures_playGeneration)
            logger_handle.log("============== starting training ================")
            train_data = mem.getTrainSamples()
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
            logger_handle.log("============ starting pit =============")
            futures_pit = [
                mcts.execute_pit.remote(current_Network_path, new_net_path, 1 if pit_iter % 2 == 0 else -1,
                                        configs.SIMS_FAKTOR,
                                        configs.SIMS_EXPONENT) for pit_iter in range(configs.EVAL_EPISODES)]
            finished, not_finished = ray.wait(futures_pit)
            oldWins = 0
            newWins = 0
            while not_finished:
                winner = ray.get(finished)[0]
                if winner == 1:
                    newWins += 1
                elif winner == -1:
                    oldWins += 1
                logger_handle.log("player won: " + str(winner))
                futures_pit = not_finished
                finished, not_finished = ray.wait(futures_pit)
            if newWins < oldWins * configs.SCORING_THRESHOLD:  # not better then previus
                logger_handle.log("falling back to old network")
                current_Network.load_weights(current_Network_path)
            else:
                logger_handle.log("new network accepted")
            if episode <= configs.MEMORY_ITERS:
                current_mem_size = int(
                    configs.MIN_MEMORY + (configs.MAX_MEMORY - configs.MIN_MEMORY) * (episode / configs.MEMORY_ITERS))
                logger_handle.log("changed mem size to " + str(current_mem_size))
                mem.changeMaxSize(current_mem_size)
            episode += 1
        logger_handle.log("============ finisched AlphaZero ===========")
        current_Network.save_weights("best_net.h5")
        mem.saveState(episode, "finished_array.npy", "finisched_vars.obj")
    except Exception as e:
        print(e)
        logger_handle.log("============ interupted training ===========")
        logger_handle.log(str(e))
        mem.saveState(episode, "interrupt_array.npy", "interrupted_vars.obj")
    finally:
        ray.shutdown()
