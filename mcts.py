import multiprocessing as mp
from collections import defaultdict

import numpy as np

from MillEnv import MillEnv


class State(object):
    def __init__(self, env: MillEnv, last_move=None, parent=None, seed=None):
        self.last_move = last_move
        self.n = 0
        self.q = 0
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
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self):
        return self.valid_moves[np.random.randint(len(self.valid_moves))]

    def is_terminal_node(self):
        return self.terminal

    def expand(self):
        action = self.untried_actions.pop()
        self.env.setFullState(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5],
                              self.state[6], self.state[7], self.state[8], self.state[9])
        self.env.makeMove(action)
        child_node = State(self.env, action, parent=self)
        self.children = np.append(self.children, [child_node])
        return child_node

    def rollout(self, discount, max_depth=20, root=None):
        if self.parent is not None:
            self.env.setFullState(self.parent.state[0], self.parent.state[1], self.parent.state[2],
                                  self.parent.state[3], self.parent.state[4], self.parent.state[5],
                                  self.parent.state[6], self.parent.state[7], self.parent.state[8],
                                  self.parent.state[9])
        else:
            self.env.setFullState(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4],
                                  self.state[5], self.state[6], self.state[7], self.state[8], self.state[9])
        reward = defaultdict(float)
        depth = max_depth - self.depth(root)
        for iteration in range(depth):
            last_player = self.env.isPlaying
            if iteration == 0 and self.last_move:
                action = self.last_move
            else:
                action = self.random_gen.choice(self.env.getValidMoves())
            if iteration == 0:
                self.move_value = self.env.makeMove(action)[1]
            else:
                reward[last_player] += self.env.makeMove(action)[1]
            if self.env.isFinished() != 0:
                depth = iteration
                break
            iteration += 1
        if self.parent:
            if self.env.isFinished() == self.parent.state[1]:
                reward[self.parent.state[1]] += discount ** depth * 10
            elif self.env.isFinished() == -self.parent.state[1]:
                reward[-self.parent.state[1]] += discount ** depth * 10
            elif self.env.isFinished() == 2:
                reward[self.parent.state[1]] -= discount ** depth * 0.5
        return reward

    def depth(self, root):
        depth = 0
        current_parent = self
        while current_parent.parent != root and current_parent.parent:
            current_parent = current_parent.parent
            depth += 1
        return depth

    def backpropagate(self, result, max_depth):
        current_node = self
        iters = 1
        while current_node.parent:
            current_node.n += 1
            result[current_node.parent.state[1]] += current_node.move_value / iters
            current_node.q += result[current_node.parent.state[1]]
            current_node.q -= result[-current_node.parent.state[1]]
            current_node = current_node.parent
            iters += 1
        current_node.n += 1

    def mergeChildren(self, states: np.ndarray):
        self.q = np.sum([state.q for state in states])
        self.n = np.sum([state.n for state in states])
        childrenAll: np.ndarray = np.array([state.children for state in states], dtype=object).transpose()
        children: np.ndarray
        for children in childrenAll:
            valididx: np.ndarray = ~(children == None)
            if len(valididx) >= 1:
                self.children = np.append(self.children, children[valididx].reshape(-1)[0].merge(children[valididx]))
        return self

    def merge(self, children):
        self.q = np.sum([state.q for state in children])
        self.n = np.sum([state.n for state in children])
        return self


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
        self.root.mergeChildren(np.array(result))
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
