import cfgs.config as config
import argparse, yaml
import random
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser(description='KeTransformer Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test'],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--model', dest='model', default="bert", type=str)

    parser.add_argument('--p', dest='pooling',
                        type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['imdb', 'yelp_13', 'yelp_14',],
                        help='{imdb, yelp_13, yelp_14}',
                        default='imdb', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    args = parser.parse_args()
    return args


if __name__ == '__main__': # python main.py --run train --dataset imdb --version ketransformer --gpu 0 --model bert --p la
    __C = config.__C

    args = parse_args()
    try:
        cfg_file = "cfgs/{}_model.yml".format(args.model)
        with open(cfg_file, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        args_dict = edict({**yaml_dict, **vars(args)})
    except:
        args_dict = edict({**vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)
    from trainer import trainer

    execution = trainer.BERTTrainer(__C)
    execution.run(__C.run_mode)