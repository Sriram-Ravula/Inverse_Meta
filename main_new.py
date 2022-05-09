import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils_new.exp_utils import set_all_seeds, parse_args, parse_config
from metalearners.gradientlearner import GBML
import sys

def main():
    args = parse_args(globals()['__doc__'])
    hparams = parse_config(args.config)

    print("\nWriting to ", os.path.join(hparams.save_dir, args.doc), '\n')

    set_all_seeds(hparams.seed)

    learner = GBML(hparams, args)
    learner.run_meta_opt()

    return 0

if __name__ == '__main__':
    sys.exit(main())