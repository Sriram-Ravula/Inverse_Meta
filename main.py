from utils.utils import parse_args, parse_config, set_all_seeds
import os
from .meta_learner import MetaLearner
import sys

def main():
    args = parse_args(globals()['__doc__'])
    hparams = parse_config(args.config)

    print("Writing to ", os.path.join(hparams.save_dir, args.doc))

    set_all_seeds(hparams.seed)

    learner = MetaLearner(hparams, args)
    learner.run_meta_opt()

    return 0

if __name__ == '__main__':
    sys.exit(main())
