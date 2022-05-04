# Inverse-Meta
Meta Learning for Inverse Problems with Generative Models.

## Structure
- **algorithms**: algorithms for solving the inner (i.e. inverse) problem
- **configs**: yaml config files for running experiments
- **datasets**: PyTorch dataset classes
- **metalearners**: the main control classes for gradient-based meta-learning
- **problems**: defines forward operators as classes for re-usability
- **utils_new**: useful functions for experiment logging, metrics, and losses
- ```main_new.py```: program to invoke for running meta-learning from command line

## Settings
Meta learning can be run in 2 modes:
- scalar: learn a hyperparameter for weighting one term in the inner solution algorithm, e.g. we want to learn scalar c in: L = c * ||Ax - y||^2 - log p(x)
- vector: learn learn a hyperparameter for weighting each individual measurement in the inner solution, e.g. learn a vector c in: L = ||Diag(c)(Ax - y)||^2 - log p(x)

Furthermore, we can learn an sampling pattern for measurements, e.g. for square matrix A we want to find the matrix C for the inner loss: L = ||C(Ax - y)||^2 - log p(x) that results in the best reconstruction. Currently, only a diagonal C can be learned for this mode - i.e. C point-wise multiplies the residual (Ax - y).

## How to run
Check **configs** > ```sample_ffhq.yml``` for an example of how to set up a configuration file for you experiment along with details for each parameter.

To start an experiment, run the command
```
python main_new.py --config <full/path/to/config/file.yml> --doc <experiment_name>
```

For knees:
```
python main_new.py --config configs/sample_ddrm_knees.yml --doc new --timesteps 1000 --eta 0.7 --etaB 1.0
```

## Submodule initialization
```
git submodule update --init --recursive
```

## Download checkpoints
Knee checkpoint:
```
gdown https://drive.google.com/uc?id=1VOrIq8_53Oy6J-t90rzZDiZnLu-oeDN8
```

Brain checkpoint:
```
gdown https://drive.google.com/uc?id=1aSjBUnce-rrtrpFS_BdARpOjOEWfY_gA
```

Knee data:
```
gdown https://drive.google.com/uc?id=1q5B8sG_5rXHNLzv3AuUQ7L4snl4Cx7kJ
```
