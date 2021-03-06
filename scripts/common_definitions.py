"""
Common definitions of variables that can be used across files
Original author - Samuel Matthew Koesnadi
"""

from tensorflow.keras.initializers import glorot_normal

# general parameters
CHECKPOINTS_PATH = "/home/lovro/sphero_ws/src/sphero_formation/training_results/DDPG/checkpoints/DDPG_"
TF_LOG_DIR = '/home/lovro/sphero_ws/src/sphero_formation/training_results/DDPG/logs/'

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()
# KERNEL_INITIALIZER = tf.random_uniform_initializer(-1.5e-3, 1.5e-3)

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 200
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 10000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
WARM_UP = 100  # num of warm up epochs
