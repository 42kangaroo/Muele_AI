import numpy as np

import configs
import encoders
from MillEnv import MillEnv


def generate_empty_nodes(child_node):
    for child_move in child_node.valid_moves:
        child_node.env.setFullState(child_node.state[0], child_node.state[1], child_node.state[2],
                                    child_node.state[3], child_node.state[4], child_node.state[5],
                                    child_node.state[6], child_node.state[7], child_node.state[8],
                                    child_node.state[9])
        child_node.env.makeMove(child_move)
        child_node.children[child_move] = State(None, 0, child_node.state[1], child_node.env, child_move,
                                                child_node)
        if abs(child_node.children[child_move].is_terminal_node()) == 1:
            child_node.children.clear()
            child_node.children[child_move] = State(None, 0, child_node.state[1], child_node.env, child_move,
                                                    child_node)
            child_node.valid_moves = [child_move]
            return True
    return False


class State(object):
    def __init__(self, priors, value, last_player, env: MillEnv, last_move=None, parent=None):
        self.last_player = last_player
        self.last_move = last_move
        self.n = 0
        self.q = value
        self.w = 0
        self.priors = priors
        self.terminal = env.isFinished()
        self.valid_moves = env.getValidMoves()
        self.state = env.getFullState()
        self.parent = parent
        self.children = {}
        self.env: MillEnv = env
        self.generator = np.random.default_rng()
        self.is_visited = False
        if not self.parent:
            generate_empty_nodes(self)

    def add_noise(self, alpha=configs.ROOT_DIRICHLET_ALPHA, weight=configs.ROOT_DIRICHLET_WEIGHT):
        noise = self.generator.dirichlet([alpha] * len(self.valid_moves))
        self.priors[0, self.valid_moves] = self.priors[0, self.valid_moves] * (1 - weight) + noise * weight

    def best_child(self, c_param=configs.CPUCT):
        choices_weights = np.full(24, -1.1)
        for idx, node in self.children.items():
            if abs(node.is_terminal_node()) == 1:
                return node
            choices_weights[idx] = node.w + c_param * self.priors[0, idx] * (np.sqrt(self.n) / (1 + node.n))
        return self.children[np.argmax(choices_weights)]

    def pi(self, temp):
        if temp == 0:
            choices_weights = np.zeros(24)
            choices_weights[max(self.children.items(), key=lambda x: x[1].n)[0]] = 1

        else:
            visitSum = sum([np.power(c.n, 1 / temp) for c in self.children.values()])
            choices_weights = np.zeros(24)
            for idx, node in self.children.items():
                choices_weights[idx] = np.power(node.n, 1 / temp) / visitSum
        return choices_weights

    def is_terminal_node(self):
        return self.terminal

    def expand(self, move, nnet):
        child_node: State = self.children[move]
        if not child_node.is_visited:
            child_node.is_visited = True
            if child_node.is_terminal_node() != 0:
                return child_node, child_node.is_terminal_node()
            val = child_node.setValAndPriors(nnet)
            if generate_empty_nodes(child_node):
                return child_node.expand(child_node.valid_moves[0], None)
            return child_node, val
        return child_node, child_node.w

    def setValAndPriors(self, nnet):
        if self.is_terminal_node() == 0:
            self.priors, val = nnet(
                encoders.prepareForNetwork([self.state[0]], [self.state[1]], [self.state[3]],
                                           [self.state[2][1 if self.env.isPlaying == 1 else 0]], [self.state[7]]))
            self.priors = self.priors.numpy()
        else:
            if self.is_terminal_node() == 2:
                val = 0
            else:
                val = self.is_terminal_node()
            self.priors = np.zeros((1, 24))
        return val

    def backpropagate(self, result):
        result = result * self.state[1]
        current_node = self
        while current_node:
            current_node.n += 1
            current_node.q += result * current_node.last_player
            current_node.w = current_node.q / current_node.n
            current_node = current_node.parent

    def discardParent(self):
        self.parent = None


class MonteCarloTreeSearch(object):
    def __init__(self, node, memory_size=configs.MIN_MEMORY, depth=0):
        self.root: State = node
        self.depth = depth
        self.memory_size = memory_size

    def generatePlay(self, memory, nnet, index, multiplikator=configs.SIMS_FAKTOR,
                     exponent=configs.SIMS_EXPONENT):
        short_term_memory = []
        for iteration in range(configs.MAX_MOVES):
            if self.root.is_terminal_node() != 0:
                break
            pi = self.search(nnet, multiplikator, exponent)
            s_states, s_selected = encoders.getSymetries(self.root.state[0], self.root.state[7])
            s_gamePhase = np.full(8, self.root.state[2][1 if self.root.last_player == 1 else 0])
            s_player = np.full(8, self.root.last_player)
            s_moveNeeded = np.full(8, self.root.state[3])
            s_pi = encoders.getTargetSymetries(pi, self.root.state[3])
            for state, player, selected, gamePhase, moveNeeded, pi_s in zip(s_states, s_player, s_selected, s_gamePhase,
                                                                            s_moveNeeded,
                                                                            s_pi):
                short_term_memory.append([state, player, selected, gamePhase, moveNeeded, pi_s, None])
            if self.depth < configs.TURNS_UNTIL_TAU0:
                self.goToMoveNode(np.random.choice(np.arange(24), p=pi))
            else:
                self.goToMoveNode(np.argmax(pi))
        print(len(short_term_memory), self.root.is_terminal_node())
        finished = self.root.is_terminal_node()
        if abs(finished) == 1:
            pass
        else:
            finished = (self.root.state[5][1] - self.root.state[5][0]) / 6
        short_term_memory = np.array(short_term_memory, dtype=object)
        short_term_memory[:, 6] = finished * short_term_memory[:, 1]
        with index.get_lock():
            idx_val = index.value
            if idx_val + len(short_term_memory) < self.memory_size:
                idxs_until_full = None
                index.value += len(short_term_memory)
            else:
                idxs_until_full = self.memory_size - idx_val
                index.value = len(short_term_memory) - idxs_until_full
        if idxs_until_full:
            memory[np.arange(idx_val, self.memory_size)] = short_term_memory[:idxs_until_full]
            memory[np.arange(len(short_term_memory) - idxs_until_full)] = short_term_memory[idxs_until_full:]
        else:
            memory[np.arange(idx_val, idx_val + len(short_term_memory))] = short_term_memory

    def search(self, nnet, multiplikator=configs.SIMS_FAKTOR, exponent=configs.SIMS_EXPONENT):
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
        simulations_number = int(len(self.root.valid_moves) ** exponent * multiplikator)
        self.root.add_noise()
        for i in range(simulations_number):
            v, reward = self._tree_policy(nnet)
            v.backpropagate(reward)
        return self.root.pi(1)

    def goToMoveNode(self, move):
        self.root = self.root.children[move]
        self.root.discardParent()
        if self.root.is_terminal_node() == 0:
            self.root.add_noise()
            self.depth += 1

    def _tree_policy(self, nnet, expl=configs.CPUCT):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """

        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_visited and current_node.parent:
                return current_node.parent.expand(current_node.last_move, nnet)
            else:
                current_node = current_node.best_child(expl)
        return current_node, current_node.w
