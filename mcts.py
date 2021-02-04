import numpy as np
import ray

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
            child_node.children[child_move] = State(None, 0, child_node.state[1], child_node.env,
                                                    child_move,
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
        self.is_visited = False
        if not self.parent:
            generate_empty_nodes(self)

    def add_noise(self, gen: np.random.Generator, alpha=configs.ROOT_DIRICHLET_ALPHA,
                  weight=configs.ROOT_DIRICHLET_WEIGHT):
        noise = gen.dirichlet([alpha] * len(self.valid_moves))
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
            choices_weights = np.full(24, -1.)
            choices_weights[max(self.children.items(), key=lambda x: x[1].n)[0]] = 1
        else:
            visitSum = sum([np.power(c.n, 1 / temp) for c in self.children.values()])
            choices_weights = np.full(24, -1.)
            for idx, node in self.children.items():
                choices_weights[idx] = np.power(node.n, 1 / temp) / visitSum
        return choices_weights

    def is_terminal_node(self):
        return self.terminal

    def expand(self, move, nnet):
        child_node: State = self.children[move]
        if not child_node.is_visited:
            child_node.is_visited = True
            if abs(child_node.is_terminal_node()) == 1:
                return child_node, child_node.is_terminal_node()
            val = child_node.setValAndPriors(nnet)
            if generate_empty_nodes(child_node):
                return child_node.expand(child_node.valid_moves[0], None)
            return child_node, val
        return child_node, child_node.w

    def setValAndPriors(self, nnet):
        if self.is_terminal_node() == 0:
            from tensorflow.keras.backend import softmax
            self.priors, val = nnet(
                encoders.prepareForNetwork([self.state[0]], [self.state[1]], [self.state[3]],
                                           [self.state[2][1 if self.state[1] == 1 else 0]], [self.state[7]]))
            mask = np.ones(self.priors.shape, dtype=bool)
            mask[0, self.valid_moves] = False
            self.priors = np.array(self.priors)
            self.priors[mask] = -100.
            self.priors = softmax(self.priors).numpy()
        else:
            if self.is_terminal_node() == 2:
                val = 0
            else:
                val = self.is_terminal_node()
            self.priors = np.zeros((1, 24))
        return val

    def backpropagate(self, result):
        if self.is_terminal_node() == 0:
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
    def __init__(self, node):
        self.root: State = node
        self.gen = np.random.default_rng()
        self.depth = 0

    def generatePlay(self, memory, nnet_weights_path, logger, multiplikator=configs.SIMS_FAKTOR,
                     exponent=configs.SIMS_EXPONENT):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        import Network
        nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                               configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
        nnet.load_weights(nnet_weights_path)
        short_term_memory = []
        val = self.root.setValAndPriors(nnet)
        self.root.backpropagate(val)
        for iteration in range(configs.MAX_MOVES):
            if self.root.is_terminal_node() != 0:
                break
            pi = self.search(nnet, multiplikator, exponent)
            s_states, s_selected = encoders.getSymetries(self.root.state[0], self.root.state[7])
            s_gamePhase = np.full(8, self.root.state[2][1 if self.root.state[1] == 1 else 0])
            s_player = np.full(8, self.root.state[1])
            s_moveNeeded = np.full(8, self.root.state[3])
            s_pi = encoders.getTargetSymetries(pi, self.root.state[3])
            for state, player, selected, gamePhase, moveNeeded, pi_s in zip(s_states, s_player, s_selected, s_gamePhase,
                                                                            s_moveNeeded,
                                                                            s_pi):
                short_term_memory.append([state, player, selected, gamePhase, moveNeeded, pi_s, None])
            if self.depth < configs.TURNS_UNTIL_TAU0:
                choices_pi = np.where(pi == -1, np.zeros(pi.shape), pi)
                self.goToMoveNode(np.random.choice(np.arange(24), p=choices_pi))
            else:
                self.goToMoveNode(np.argmax(pi))
        logger.log.remote(
            "turns played: " + str(len(short_term_memory) // 8) + " player won: " + str(self.root.is_terminal_node()))
        finished = self.root.is_terminal_node()
        if abs(finished) == 1:
            pass
        else:
            finished = (self.root.state[5][1] - self.root.state[5][0]) / 6
        short_term_memory = np.array(short_term_memory, dtype=object)
        short_term_memory[:, 6] = finished * short_term_memory[:, 1]
        memory.addToMem.remote(short_term_memory)

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
        if not self.root.is_visited:
            val = self.root.setValAndPriors(nnet)
            self.root.backpropagate(val)
        self.root.add_noise(self.gen)
        for i in range(simulations_number):
            v, reward = self._tree_policy(nnet)
            v.backpropagate(reward)
        return self.root.pi(1)

    def goToMoveNode(self, move):
        if len(self.root.children) == 0:
            generate_empty_nodes(self.root)
        self.root = self.root.children[move]
        if len(self.root.children) == 0:
            generate_empty_nodes(self.root)
        self.root.discardParent()
        self.depth += 1

    def _tree_policy(self, nnet, expl=configs.CPUCT):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """

        current_node = self.root
        while True:
            if not current_node.is_visited and current_node.parent:
                return current_node.parent.expand(current_node.last_move, nnet)
            else:
                current_node = current_node.best_child(expl)
            if current_node.is_terminal_node() != 0:
                return current_node.parent.expand(current_node.last_move, None)


@ray.remote(max_calls=1)
def execute_generate_play(memory, nnet_weights_path, logger, multiplikator=configs.SIMS_FAKTOR,
                          exponent=configs.SIMS_EXPONENT):
    env = MillEnv()
    mcts = MonteCarloTreeSearch(State(np.zeros((1, 24)), 0, -env.isPlaying, env))
    mcts.generatePlay(memory, nnet_weights_path, logger, multiplikator, exponent)
    return True


@ray.remote(max_calls=1)
def execute_pit(oldNet_path, newNet_path, begins, logger, multiplikator=configs.SIMS_FAKTOR,
                exponent=configs.SIMS_EXPONENT):
    return pit(oldNet_path, newNet_path, begins, logger, multiplikator, exponent)


def pit(oldNet_path, newNet_path, begins, logger, multiplikator=configs.SIMS_FAKTOR,
        exponent=configs.SIMS_EXPONENT):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import Network
    oldNet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    newNet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                             configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
    oldNet.load_weights(oldNet_path)
    newNet.load_weights(newNet_path)
    envNew = MillEnv()
    envOld = MillEnv()
    oldNet_mcts = MonteCarloTreeSearch(State(np.zeros((1, 24)), 0, -envOld.isPlaying, envOld))
    newNet_mcts = MonteCarloTreeSearch(State(np.zeros((1, 24)), 0, -envNew.isPlaying, envNew))
    val = oldNet_mcts.root.setValAndPriors(oldNet)
    oldNet_mcts.root.backpropagate(val)
    val = newNet_mcts.root.setValAndPriors(newNet)
    newNet_mcts.root.backpropagate(val)
    finisched = 0
    actualPlayer = 1
    for iteration in range(configs.MAX_MOVES):
        if finisched != 0:
            logger.log.remote(
                "turns played: " + str(iteration) + " player won: " + str(
                    finisched * begins))
            if finisched == 2:
                return 0
            return finisched * begins
        if actualPlayer * begins == 1:
            actualnet = newNet
            actualMCTS = newNet_mcts
        else:
            actualnet = oldNet
            actualMCTS = oldNet_mcts
        pi = actualMCTS.search(actualnet, multiplikator, exponent)
        if iteration < configs.TURNS_UNTIL_TAU0:
            choices_pi = np.where(pi == -1, np.zeros(pi.shape), pi)
            pos = np.random.choice(np.arange(24), p=choices_pi)
        else:
            pos = np.argmax(pi)
        oldNet_mcts.goToMoveNode(pos)
        newNet_mcts.goToMoveNode(pos)
        actualPlayer = actualMCTS.root.state[1]
        finisched = actualMCTS.root.is_terminal_node()
    logger.log.remote("the game ended in a draw")
    return 0
