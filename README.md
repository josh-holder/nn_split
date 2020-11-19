# nn-split
Utilizing neural networks to obtain an optimal playing strategy for SPL-T (https://simogo.com/work/spl-t/).

Core.py houses the architecture for playing simulated games of SPL-T

splt_mcts.py houses the functions to implemented Monte Carlo Tree Search

splt_model.py houses the architecture of the neural network

nn_config.py houses the neural network and simulation parameters.

nn_run.py implements the entire learning process.

USAGE: nn_run.py -r <name of folder to save runs to>
