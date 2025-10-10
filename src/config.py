"""
Configuration file for Deep Learning Assignment 1
"""

# Set to True to run the model, False to skip
config = {
    'run_knn': False,
    'run_linear': False,
    'run_nn': True,
}

# Random seed to reproduce our results
SEED = 42

# Data config
# Image resolution (28x28)
DATA_SIZE = 28
# Number of training samples to use (None for all)
SUBSAMPLE_TRAIN = None
# Number of validation samples for NN (None for all)
SUBSAMPLE_VAL = None

# Output config
PLOTS_DIR = "../plots"

# Class names
CLASS_NAMES = [
    'Basop',
    'Eosin',
    'Eryth',
    'Immat',
    'Lymph',
    'Monoc',
    'Neutr',
    'Plate',
]

# kNN hyperparameters for grid search
KNN_K_VALUES = [1, 3, 5, 7, 13, 21, 33, 45, 55, 67, 79, 91, 111, 131, 149, 167, 193, 201]
KNN_DISTANCE_METRICS = ['L2', 'L1']

# Linear classifier hyperparameters for grid search
LINEAR_LEARNING_RATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
LINEAR_REGULARIZATIONS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LINEAR_NUM_ITERS = 200
LINEAR_BATCH_SIZE = 1000
LINEAR_PRINT_EVERY = 10

# Neural network hyperparameters for grid search
NN_HIDDEN_SIZES = [50, 100, 200]
NN_LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
NN_REGULARIZATIONS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
NN_OPTIMIZERS = ['sgd', 'momentum', 'adam']
NN_NUM_ITERS = 200
NN_BATCH_SIZE = 256
NN_PRINT_EVERY = 100