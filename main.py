import gc
import multiprocessing as mp
import os
from functools import partial
from os.path import isfile
from shutil import copy

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


def execute_generate_play(nnet_path, multiplikator=configs.SIMS_FAKTOR,
                          exponent=configs.SIMS_EXPONENT):
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    nnet = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                           configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
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


def execute_pit(oldNet_path, newNet_path, begins, multiplikator=configs.SIMS_FAKTOR,
                exponent=configs.SIMS_EXPONENT):
    gc.collect()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    oldNet = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
    newNet = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
    oldNet.load_weights(oldNet_path)
    newNet.load_weights(newNet_path)
    winner = mcts.pit(oldNet, newNet, begins, multiplikator, exponent)
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    del oldNet
    del newNet
    gc.collect()
    return winner


def train_net(in_path, out_path, train_data, tensorboard_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    tensorboard_callback = keras.callbacks.TensorBoard(tensorboard_path, update_freq=10,
                                                       profile_batch=2)
    current_Network = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                      configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
    current_Network.load_weights(in_path)
    current_Network.compile(optimizer='adam',
                            loss={'policy_output': Network.cross_entropy_with_logits, 'value_output': 'mse'},
                            loss_weights=[0.5, 0.5],
                            metrics=['accuracy'])
    current_Network.fit(
        encoders.prepareForNetwork(train_data[0], train_data[1], train_data[2], train_data[3],
                                   train_data[4]),
        {'policy_output': train_data[5],
         'value_output': train_data[6]}, epochs=configs.EPOCHS,
        batch_size=configs.BATCH_SIZE, callbacks=[tensorboard_callback], shuffle=True)
    current_Network.save_weights(out_path)


def save_first_net(path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("hello")
    current_Network = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                      configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
    print(current_Network.summary())
    current_Network.save_weights(path)


def save_whole_net(weights_path, model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    current_Network = Network.get_net(configs.FILTERS, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                      configs.NUM_ACTIONS, configs.INPUT_SIZE, None, configs.NUM_RESIDUAL)
    current_Network.load_weights(weights_path)
    current_Network.save(model_path)


if __name__ == "__main__":
    logger_handle = logger.Logger(configs.LOGGER_PATH)
    current_mem_size = configs.MIN_MEMORY
    mem = memory.Memory(current_mem_size, configs.MAX_MEMORY)
    episode = 0
    init_net_p = mp.Process(target=save_first_net, args=(configs.BEST_PATH,))
    init_net_p.start()
    init_net_p.join()
    del init_net_p
    gc.collect()
    copy("configs.py", configs.INTERMEDIATE_SAVE_PATH + "configs.py")
    copy(configs.BEST_PATH, configs.NEW_NET_PATH)
    if isfile("interrupt_array.npy") and isfile("interrupted_vars.obj"):
        episode = mem.loadState("interrupt_array.npy", "interrupted_vars.obj")
        copy(configs.NETWORK_PATH + str(episode) + ".h5", configs.BEST_PATH)
        logger_handle.log("=========== restarting training ==========")
    try:
        while episode <= configs.TRAINING_LOOPS:
            logger_handle.log("============== starting episode " + str(episode) + " ===============")
            current_Network_path = configs.NETWORK_PATH + str(episode) + ".h5"
            copy(configs.BEST_PATH, current_Network_path)
            logger_handle.log("saving actual net to " + current_Network_path)
            logger_handle.log("saving intermediate arrays")
            mem.saveState(episode, configs.INTERMEDIATE_SAVE_PATH + "interrupt_array.npy",
                          configs.INTERMEDIATE_SAVE_PATH + "interrupted_vars.obj")
            logger_handle.log("============== starting selfplay ================")
            with mp.Pool(configs.NUM_CPUS, maxtasksperchild=10) as pool:
                for stmem in pool.imap_unordered(partial(execute_generate_play, configs.BEST_PATH, configs.SIMS_FAKTOR),
                                                 [
                                                     configs.SIMS_EXPONENT for play in
                                                     range(configs.EPISODES)]):
                    mem.addToMem(stmem)
                    logger_handle.log("player won: " + str(stmem[0][6]) + " turns played: " + str(len(stmem) // 8))
                    del stmem
                    gc.collect()
            del pool
            gc.collect()
            logger_handle.log("saving intermediate arrays")
            mem.saveState(episode, configs.INTERMEDIATE_SAVE_PATH + "interrupt_array.npy",
                          configs.INTERMEDIATE_SAVE_PATH + "interrupted_vars.obj")
            logger_handle.log("============== starting training ================")
            train_data = mem.getTrainSamples()
            train_p = mp.Process(
                target=train_net, args=(configs.NEW_NET_PATH, configs.NEW_NET_PATH, train_data,
                                        configs.TENSORBOARD_PATH + str(episode)))
            train_p.start()
            train_p.join()
            del train_p
            gc.collect()
            logger_handle.log("============ starting pit =============")
            oldWins = 0
            newWins = 0
            with mp.Pool(configs.NUM_CPUS, maxtasksperchild=10) as pool:
                for win in pool.imap_unordered(
                        partial(execute_pit, configs.BEST_PATH, configs.NEW_NET_PATH, exponent=configs.SIMS_EXPONENT,
                                multiplikator=configs.SIMS_FAKTOR),
                        [1 if pit_iter % 2 == 0 else -1 for pit_iter in range(configs.EVAL_EPISODES)]):
                    if win == 1:
                        newWins += 1
                    elif win == -1:
                        oldWins += 1
                    logger_handle.log("player won: " + str(win))
            del pool
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
        save_p = mp.Process(target=save_whole_net,
                            args=(configs.BEST_PATH, configs.INTERMEDIATE_SAVE_PATH + "models/whole_net"))
        save_p.start()
        save_p.join()
        mem.saveState(episode, "finished_array.npy", "finisched_vars.obj")
    except BaseException as e:
        print(e.__doc__)
        print(e)
        logger_handle.log("============ interupted training ===========")
        logger_handle.log(str(e.__doc__))
        logger_handle.log(str(e))
        save_p = mp.Process(target=save_whole_net,
                            args=(configs.BEST_PATH, configs.INTERMEDIATE_SAVE_PATH + "models/whole_net"))
        save_p.start()
        save_p.join()
        mem.saveState(episode, "interrupt_array.npy", "interrupted_vars.obj")
