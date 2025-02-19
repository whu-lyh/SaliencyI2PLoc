import argparse
import os
from pathlib import Path


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not args.test:
        if not os.path.exists(args.tfboard_path):
            os.makedirs(args.tfboard_path, exist_ok=True)
            print('Create TFBoard path successfully at %s' % args.tfboard_path)


def get_args():
    '''
        Load all parameters from yaml file
    '''
    parser = argparse.ArgumentParser(description='Cross-modality retrieval task')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local process id for DDP training, this will be automatically set by os.environ["LOCAL_RANK"]')
    parser.add_argument('--num_workers', type=int, default=12, help='equal to the number of threads')
    parser.add_argument('--gpu', type=int, default=0, help='choose single GPU id to use, otherwise, do not adjust this')
    parser.add_argument('--debug', action='store_true', help="debug mode")
    # seed 
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend')
    # some args
    parser.add_argument('--refactor', action='store_true', default=False, help='set whole process with train process refacted.')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='validate freq')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--clip_grad', type=bool, default=True, help='clips gradient norm')
    parser.add_argument('--max_grad_norm', type=float, default=0.9, help="max gradient norm")
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError('ckpts should not be None while test mode')

    if 'LOCAL_RANK' not in os.environ: # customize the local rank id, do not adjust this manually, which is already deprecated by Pytorch officially
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = args.exp_name + '_test'
    else:
        args.exp_name = args.exp_name + '_train_val'
        
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, 'TFBoard', args.exp_name)
    args.log_name = Path(args.config).stem
    args.wandb_path = os.path.join('./experiments', Path(args.config).stem)
    create_experiment_dir(args)
    return args