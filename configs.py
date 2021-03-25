from tensorflow import convert_to_tensor

# SELF PLAY
MAX_MOVES = 280
ROOT_DIRICHLET_ALPHA = 0.8
ROOT_DIRICHLET_WEIGHT = 0.25
TURNS_UNTIL_TAU0 = 10
CPUCT = 3
SIMS_FAKTOR = 12
SIMS_EXPONENT = 1.05
EPISODES = 180
MIN_MEMORY = 150000
MAX_MEMORY = 500000
MEMORY_ITERS = 15

# RETRAINING

BATCH_SIZE = 128
EPOCHS = 12
FILTERS = 128
OUT_FILTERS = 128
HIDDEN_SIZE = 256
INPUT_SIZE = 24, 4
NUM_ACTIONS = 76
NUM_RESIDUAL = 4
NETWORK_PATH = "run5/models/episode-"
BEST_PATH = "run5/models/best_net.h5"
NEW_NET_PATH = "run5/models/new_net.h5"
TENSORBOARD_PATH = "run5/Tensorboard/episode-"

# EVAL

EVAL_EPISODES = 24
SCORING_THRESHOLD = 1.17

# GENERAL
NUM_CPUS = 12
TRAINING_LOOPS = 50
LOGGER_PATH = "run5/AlphaLog.log"
INTERMEDIATE_SAVE_PATH = "run5/"
FILTERS_ARRAY = convert_to_tensor([[[1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 1., 0., 0.],
                                    [1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.],
                                    [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 1.],
                                    [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                     0., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                     0., 0., 0., 0., 1., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1.,
                                     0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
                                     0., 1., 0., 0., 0., 0., 0., 0.],
                                    [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
                                     0., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
                                     0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0.,
                                     0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
                                     0., 0., 0., 0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
                                     0., 0., 0., 0., 0., 0., 0., 1.],
                                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
                                     1., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                     1., 1., 0., 1., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                                     1., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                     0., 0., 1., 1., 1., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     1., 0., 1., 1., 1., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                     0., 0., 1., 1., 1., 0., 0., 0.],
                                    [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                     0., 0., 0., 0., 0., 1., 1., 1.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                     1., 0., 0., 1., 0., 1., 1., 1.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                     0., 0., 0., 0., 0., 1., 1., 1.]]])
