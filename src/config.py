"""
Configuration file for Deep Learning Assignment 1
"""

# Set to True to run the model, False to skip
config = {
    'run_knn': True,        # k-Nearest Neighbors classifier
    'run_linear': True,     # Linear classifiers (SVM and Softmax)
    'run_nn': True,         # Neural Network
}

# Random seed to reproduce our results
SEED = 42

# Data config
DATA_SIZE = 28              # Image resolution (28x28)
SUBSAMPLE_TRAIN = 5000      # Number of training samples to use (None for all)
SUBSAMPLE_VAL = 500         # Number of validation samples for NN (None for all)

# Output config
PLOTS_DIR = "../plots"      # Where to save plots

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
LINEAR_LEARNING_RATES = [1e-4, 5e-4, 1e-3]
LINEAR_REGULARIZATIONS = [1e-4, 5e-4, 1e-3, 5e-3]
LINEAR_NUM_ITERS = 200
LINEAR_BATCH_SIZE = 200
LINEAR_PRINT_EVERY = 100

# Neural network hyperparameters for grid search
NN_HIDDEN_SIZES = [100, 200, 500]
NN_LEARNING_RATES = [0.001, 0.005, 0.01]
NN_REGULARIZATIONS = [0.0001, 0.001, 0.01]
NN_OPTIMIZERS = ['sgd', 'momentum', 'adam']
NN_NUM_ITERS = 200
NN_BATCH_SIZE = 64
NN_PRINT_EVERY = 100