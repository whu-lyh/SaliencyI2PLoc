
import torch

from datasets.build import build_dataset_from_cfg
from tools import builder
from tools.validate import validate
from utils.logger import *


def test(args, config):
    '''
        test the pretrained model, multi sequences are supported
    '''
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    # build dataset
    print_log("Loading testing datasets ...", logger=logger)
    test_dataset = build_dataset_from_cfg(config.dataset.test._base_, config.dataset.test.others)
    print_log(test_dataset.get_dataset_info(), logger=logger)
    print_log("Loading model ...", logger=logger)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    epoch_num = builder.load_model(base_model, args.ckpts, logger=logger)
    # No DDP inplementation
    if args.gpu is not None:
        # do not use all gpus
        torch.cuda.set_device(args.gpu)
        base_model = base_model.cuda(args.gpu)
        print_log("Not using data parallel ...", logger=logger)
    else:
        print_log("CUDA device detected: " + str(torch.cuda.device_count()), logger=logger)
        if torch.cuda.device_count() > 1:
            # DataParallel will divide and allocate batch_size to all available GPUs
            base_model = torch.nn.DataParallel(base_model).cuda()
            print_log("Using data parallel ...", logger=logger)
    print_log("Model loading finished ...", logger=logger)
    # core test function
    _ = validate(base_model=builder.unwrap_dist_model(args, base_model), base_dataset=test_dataset, 
                 args=args, config=config, logger=logger, writer=None, epoch_num=epoch_num, write_tboard=False)