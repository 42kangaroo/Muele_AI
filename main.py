import gc
import os
from os.path import isfile
from shutil import copy

import ray
import tensorflow as tf
from numpy import zeros
from tensorflow import keras

import MillEnv
import Network
import configs
import encoders
import logger
import mcts
import memory


@ray.remote(num_cpus=1, num_gpus=0)
def execute_generate_play(nnet_path, multiplikator=configs.SIMS_FAKTOR,
                          exponent=configs.SIMS_EXPONENT):
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    nnet.load_weights(nnet_path)
    env = MillEnv.MillEnv()
    mcts_ = mcts.MonteCarloTreeSearch(mcts.State(zeros((1, 24)), 0, -env.isPlaying, env))
    stmem = mcts_.generatePlay(nnet, multiplikator, exponent)
    del mcts_
    del env
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del nnet
    gc.collect()
    return stmem


@ray.remote(num_cpus=1, num_gpus=0)
def execute_pit(oldNet_path, newNet_path, begins, multiplikator=configs.SIMS_FAKTOR,
                exponent=configs.SIMS_EXPONENT):
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    oldNet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    newNet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    oldNet.load_weights(oldNet_path)
    newNet.load_weights(newNet_path)
    winner = mcts.pit(oldNet, newNet, begins, multiplikator, exponent)
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del oldNet
    del newNet
    gc.collect()
    return winner


if __name__ == "__main__":
    ray.shutdown()
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
    current_Network.save_weights(configs.BEST_PATH)
    del current_Network
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    try:
        while episode <= configs.TRAINING_LOOPS:
            logger_handle.log("============== starting episode " + str(episode) + " ===============")
            current_Network_path = configs.NETWORK_PATH + str(episode) + ".h5"
            copy(configs.BEST_PATH, current_Network_path)
            logger_handle.log("saving actual net to " + current_Network_path)
            logger_handle.log("============== starting selfplay ================")
            ray.init()
            futures_playGeneration = [
                execute_generate_play.remote(configs.BEST_PATH, configs.SIMS_FAKTOR,
                                             configs.SIMS_EXPONENT) for play in range(configs.EPISODES)]

            while True:
                finished, not_finished = ray.wait(futures_playGeneration)
                gc.collect()
                stmem = ray.get(finished)[0]
                mem.addToMem(stmem)
                logger_handle.log("player won: " + str(-stmem[0][6]) + " turns played: " + str(len(stmem) // 8))
                futures_playGeneration = not_finished
                if not not_finished:
                    break
            ray.shutdown()
            del stmem
            del finished
            gc.collect()
            logger_handle.log("============== starting training ================")
            train_data = mem.getTrainSamples()
            tensorboard_callback = keras.callbacks.TensorBoard(configs.TENSORBOARD_PATH + str(episode), update_freq=10,
                                                               profile_batch=0)
            current_Network = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE,
                                              configs.OUT_FILTERS,
                                              configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
            current_Network.compile(optimizer='adam',
                                    loss={'policy_output': Network.cross_entropy_with_logits, 'value_output': 'mse'},
                                    loss_weights=[0.5, 0.5],
                                    metrics=['accuracy'])
            current_Network.fit(
                tf.convert_to_tensor(
                    encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3],
                                               train_data[4])),
                {'policy_output': tf.convert_to_tensor(train_data[5]),
                 'value_output': tf.convert_to_tensor(train_data[6])}, epochs=configs.EPOCHS,
                batch_size=configs.BATCH_SIZE, callbacks=[tensorboard_callback])
            current_Network.save_weights(configs.NEW_NET_PATH)
            del train_data
            del tensorboard_callback
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            del current_Network
            gc.collect()
            logger_handle.log("============ starting pit =============")
            ray.init()
            futures_pit = [
                execute_pit.remote(current_Network_path, configs.NEW_NET_PATH, 1 if pit_iter % 2 == 0 else -1,
                                   configs.SIMS_FAKTOR,
                                   configs.SIMS_EXPONENT) for pit_iter in range(configs.EVAL_EPISODES)]
            oldWins = 0
            newWins = 0
            while True:
                finished, not_finished = ray.wait(futures_pit)
                gc.collect()
                winner = ray.get(finished)[0]
                if winner == 1:
                    newWins += 1
                elif winner == -1:
                    oldWins += 1
                logger_handle.log("player won: " + str(winner))
                futures_pit = not_finished
                if not not_finished:
                    break
            ray.shutdown()
            del finished
            del winner
            gc.collect()
            if newWins < oldWins * configs.SCORING_THRESHOLD:  # not better then previus
                logger_handle.log("falling back to old network")
            else:
                logger_handle.log("new network accepted")
                copy(configs.NEW_NET_PATH, configs.BEST_PATH)
            if episode <= configs.MEMORY_ITERS:
                current_mem_size = int(
                    configs.MIN_MEMORY + (configs.MAX_MEMORY - configs.MIN_MEMORY) * (episode / configs.MEMORY_ITERS))
                logger_handle.log("changed mem size to " + str(current_mem_size))
                mem.changeMaxSize(current_mem_size)
            episode += 1
        logger_handle.log("============ finisched AlphaZero ===========")
        mem.saveState(episode, "finished_array.npy", "finisched_vars.obj")
    except BaseException as e:
        print(e.__doc__)
        logger_handle.log("============ interupted training ===========")
        logger_handle.log(str(e.__doc__))
        mem.saveState(episode, "interrupt_array.npy", "interrupted_vars.obj")
    finally:
        ray.shutdown()
