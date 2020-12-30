import multiprocessing as mp
from collections import defaultdict

import numpy as np

from MillEnv import MillEnv


class State(object):
    def __init__(self, priors, env: MillEnv, last_move=None, parent=None, seed=None):
        self.last_move = last_move
        self.n = 0
        self.q = 0
        self.w = 0
        self.priors = priors
        self.terminal = env.isFinished()
        self.valid_moves = env.getValidMoves()
        self.untried_actions: list = list(self.valid_moves)
        self.state = env.getFullState()
        self.parent = parent
        self.children = np.array([])
        self.results = defaultdict(float)
        self.env: MillEnv = env
        self.move_value = 0
        self.random_gen = np.random.Generator(np.random.PCG64(seed))

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.):
        choices_weights = [
            c.w + c_param * self.priors[c.last_move] * (np.sqrt(self.n) / (1 + c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def is_terminal_node(self):
        return self.terminal

    def expand(self):
        action = self.untried_actions.pop()
        self.env.setFullState(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5],
                              self.state[6], self.state[7], self.state[8], self.state[9])
        self.env.makeMove(action)
        child_node = State(1, self.env, action, parent=self)
        self.children = np.append(self.children, [child_node])
        return child_node

    def depth(self, root):
        depth = 0
        current_parent = self
        while current_parent.parent != root and current_parent.parent:
            current_parent = current_parent.parent
            depth += 1
        return depth

    def backpropagate(self, result, max_depth):
        current_node = self
        iters = 2
        while current_node.parent:
            current_node.n += 1
            result[current_node.parent.state[1]] += current_node.move_value / iters
            current_node.q += result[current_node.parent.state[1]]
            current_node.q -= result[-current_node.parent.state[1]]
            current_node = current_node.parent
            iters += 1
        current_node.n += 1


class MonteCarloTreeSearch(object):
    def __init__(self, node):
        self.root: State = node

    def best_action(self, gamma, parallel=mp.cpu_count() // 2, multiplikator=250, max_depth=20) -> int:
        """
        get best action of state
        Parameters
        -------
        :parameter gamma: the discount factor
        :parameter max_depth: depth to search actions
        :parameter multiplikator : number of simulations performed to get the best action
        Returns
        -------
        :returns best child state
        """

        with mp.Pool(parallel) as pool:
            result = pool.starmap(self.train, [(gamma, multiplikator, max_depth, i) for i in range(parallel)])
        # to select best child go for exploitation only
        self.root.env.setFullState(self.root.state[0], self.root.state[1], self.root.state[2], self.root.state[3],
                                   self.root.state[4], self.root.state[5], self.root.state[6], self.root.state[7],
                                   self.root.state[8], self.root.state[9])
        return self.root.best_child(c_param=0.).last_move

    def train(self, gamma, multiplikator=500, max_depth=20, seed=None):
        """
        get best action of state
        Parameters
        -------
        :parameter gamma: the discount factor
        :parameter max_depth: depth to search actions
        :parameter multiplikator : number of simulations performed to get the best action
        Returns
        -------
        :returns best child state
        """
        self.root.random_gen = np.random.Generator(np.random.PCG64(seed=seed))
        simulations_number = int(len(self.root.valid_moves) ** 1.2 * multiplikator)
        for i in range(simulations_number):
            v = self._tree_policy(max_depth)
            reward = v.rollout(gamma, max_depth, self.root)
            v.backpropagate(reward, max_depth)
        for child in self.root.children:
            child.children = np.array([])
        return self.root

    def setNewRoot(self, node):
        self.root = node

    def _tree_policy(self, max_depth=20, expl=0.5):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """

        current_node = self.root
        itern = 0
        while not current_node.is_terminal_node() and itern < max_depth:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child(expl)
            itern += 1
        return current_node
