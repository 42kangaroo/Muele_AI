# SELF PLAY
MAX_MOVES = 250
ROOT_DIRICHLET_ALPHA = 0.8
ROOT_DIRICHLET_WEIGHT = 0.25
TURNS_UNTIL_TAU0 = 10
CPUCT = 5
SIMS_FAKTOR = 8
SIMS_EXPONENT = 1.15
EPISODES = 120
MIN_MEMORY = 300000
MAX_MEMORY = 900000
MEMORY_ITERS = 12

# RETRAINING

BATCH_SIZE = 256
EPOCHS = 8
KERNEL_SIZE = 3
FILTERS = 96
OUT_FILTERS = 24
OUT_KERNEL_SIZE = 1
HIDDEN_SIZE = 256
INPUT_SIZE = 8, 3, 4
NUM_ACTIONS = 24
NETWORK_PATH = "models/episode-"
TENSORBOARD_PATH = "Tensorboard/episode-"

# EVAL

EVAL_EPISODES = 24
SCORING_THRESHOLD = 2

# GENERAL

TRAINING_LOOPS = 30
LOGGER_PATH = "AlphaLog.log"
