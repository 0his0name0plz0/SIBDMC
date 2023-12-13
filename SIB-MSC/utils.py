import argparse
import paddle

def cuda(tensor, uses_cuda):
    if isinstance(tensor, paddle.Tensor):
        return tensor.cuda() if uses_cuda else tensor
    return tensor.to("gpu") if uses_cuda else tensor

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



