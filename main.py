
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from tools.test import test
from tools.train_refacted import train_refacted
from utils import misc, parser
from utils.config import *
from utils.logger import *

torch.set_float32_matmul_precision('medium')# | 'high')

def main():
    # args loading
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True # Provides a speedup
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'validation'))
        else:
            train_writer = None
            val_writer = None
    # define the wandb
    wandber = None
    # config, get further configuration files of dataset and model setting
    config = get_config(args, logger=logger)
    # batch size
    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = 1
    config.dataset.test.others.bs = 1
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # set random seeds which is compatible to DDP mode
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    # run
    if args.test:
        logger.info("==> Testing <==")
        test(args, config)
        logger.info("==> Testing Finished <==")
    elif args.refactor:
        logger.info("==> Train/validating with train process refactored <==")
        train_refacted(args, config, train_writer, val_writer, wandber)
        logger.info("==> Train/validating Finished with train process refactored <==")


if __name__ == '__main__':
    main()