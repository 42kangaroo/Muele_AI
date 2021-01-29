import pickle

import numpy as np
import ray


@ray.remote(num_cpus=0)
class Memory(object):
    def __init__(self, actual_size, max_size):
        self.index = 0
        self.memory = np.zeros((max_size, 7), dtype=object)
        self.actual_max = actual_size
        self.max_size = max_size
        self.isFull = False
        self.newest_data_after_resize = 0

    def addToMem(self, short_term_mem):
        idx_val = self.index
        if self.index + len(short_term_mem) < self.actual_max:
            self.index += len(short_term_mem)
            self.memory[np.arange(idx_val, idx_val + len(short_term_mem))] = short_term_mem
        else:
            self.isFull = True
            idxs_until_full = self.actual_max - self.index
            self.index = len(short_term_mem) - idxs_until_full + self.newest_data_after_resize
            self.memory[np.arange(idx_val, self.actual_max)] = short_term_mem[:idxs_until_full]
            self.memory[np.arange(self.newest_data_after_resize, self.index)] = short_term_mem[idxs_until_full:]
            self.newest_data_after_resize = 0

    def getTrainSamples(self):
        if self.isFull:
            toFull = self.actual_max
        else:
            toFull = self.index
        policy_out, board_in = np.zeros((toFull, 24), dtype=np.float32), np.zeros((toFull, 8, 3),
                                                                                  dtype=np.float32)
        for idx in range(toFull):
            policy_out[idx], board_in[idx] = self.memory[idx, 5], self.memory[idx, 0]
        return (board_in, self.memory[:toFull, 1], self.memory[:toFull, 4],
                self.memory[:toFull, 3], self.memory[:toFull, 2], policy_out,
                self.memory[:toFull, 6].astype(np.float32))

    def changeMaxSize(self, newSize):
        if newSize > self.max_size:
            newSize = self.max_size
        if self.isFull:
            self.isFull = False
            self.newest_data_after_resize = self.index
            self.index = self.actual_max
        self.actual_max = newSize

    def saveState(self, episode, path_mem, path_vars):
        np.save(path_mem, self.memory)
        with open(path_vars, "wb") as f:
            pickle.dump(
                (episode, self.index, self.newest_data_after_resize, self.isFull, self.max_size, self.actual_max), f)

    def loadState(self, path_mem, path_vars):
        self.memory = np.load(path_mem, allow_pickle=True)
        with open(path_vars, "rb") as f:
            episode, self.index, self.newest_data_after_resize, self.isFull, self.max_size, self.actual_max = \
                pickle.load(f)
        return episode
