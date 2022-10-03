import os
import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddrm_mri'))

from utils_new.exp_utils import set_all_seeds, parse_args, parse_config
from metalearners.gradientlearner import GBML


def main():
    args = parse_args(globals()['__doc__'])
    hparams = parse_config(args.config)

    print("\nWriting to ", os.path.join(hparams.save_dir, args.doc), '\n')

    set_all_seeds(hparams.seed)

    learner = GBML(hparams, args)

    if args.test:
        learner.test()
    else:
        learner.run_meta_opt()

    return 0

if __name__ == '__main__':
    sys.exit(main())
