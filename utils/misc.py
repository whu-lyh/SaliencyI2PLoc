
import random
import re
import numpy as np


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def is_seq_of(seq, expected_type, seq_type=None):
    """
        Check whether it is a sequence of some type.
        Args:
            seq (Sequence): The sequence to be checked.
            expected_type (type): Expected type of sequence items.
            seq_type (type, optional): Expected sequence type.
        Returns:
            bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def set_random_seed(seed, deterministic=False):
    """Set random seed.
        Args:
            seed (int): Seed to be used.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Default: False.

        # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic:  # slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:  # faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark = True
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def is_url(input_url):
    """
        Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url
   
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def humanbytes(B):
    """
        Return the given bytes as a human friendly KB, MB, GB, or TB string
    """
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)

def get_flops(model, input_shape):
    from ptflops import get_model_complexity_info
    from ptflops.flops_counter import flops_to_string, params_to_string
    flops, params = get_model_complexity_info(model, input_shape, 
                                              as_strings=False, print_per_layer_stat=True, verbose=True)
    _, H, W = input_shape
    def sa_flops(h,w,dim):
        return 2*h*w*h*w*dim
    SA_flops = sa_flops(H//16, W//16, model.embed_dim)*len(model.blocks)
    flops += SA_flops   
    return flops_to_string(flops), params_to_string(params)

#if __name__ == '__main__':
    #model = SAIG_Deep()
    #flops,params = get_flops(model, (3, 224, 224))
    #print('flops: ', flops, 'params: ', params)