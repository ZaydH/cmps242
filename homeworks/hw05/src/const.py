
# Training Parameters
LEARNING_RATE = 1.0
NUM_EPOCHS = 10000
EPOCHS_PER_DECAY = 10
DECAY_RATE = 0.75

# Columns in the Pandas dataframe.
COL_HANDLE = "handle"
COL_TWEET = "tweet"
COL_TARGET = "target"
COL_TWEET_TRANSFORM = "int_transform"
COL_ONE_HOT = "one_hot"
COL_BAG_WORDS = "bag_of_words"

# Twitter handle for the two classes
HDL_DONALD_TRUMP = "realDonaldTrump"
HDL_HILLARY_CLINTON = "HillaryClinton"

# Target labels
LBL_DONALD_TRUMP = [1, 0]
LBL_HILLARY_CLINTON = [0, 1]

# Embedding Matrix Size
EMBEDDING_RANK = 50

# RNN CONSTANTS
HIDDEN_SIZE = 64
#HIDDEN_SIZE = EMBEDDING_RANK
#WORD_SIZE = 100
#RNN_OUTPUT_SIZE = 10
BATCH_SIZE = -1

NUM_NEURON_HIDDEN_LAYER = 256
NUM_HIDDEN_LAYERS = 1
