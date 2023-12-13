import argparse
import random

import numpy as np
# import torch
import paddle
# from torch.backends import cudnn
from solver import Solver
from solver_ft import Solver_ft
from utils import str2bool

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    paddle.seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)

    paddle.fluid.core.globals()['FLAGS_cudnn_deterministic'] = True

    print(' ft:', args.ft, ' gamma-MLP:', args.gamma, ' alpha-DIS:', args.alpha, ' mKL:', args.mkl, ' Cmi:', args.cmi, ' rec:', args.rec, ' c2', args.c2, ' selfExp:', args.selfExp, ' ckpt_dir-save_name:', args.save_name)
    print('batch_size: ', args.batch_size)
    if not args.ft:
        print("---------- train ----------")
        net = Solver(args)
        net.train()

    else:
        print("---------- finetune ----------")
        net_ft = Solver_ft(args)
        net_ft.finetune()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-supervised Information Bottleneck for Deep Multi-view Subspace Clustering')

    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')

    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--ft', default=False, type=str2bool, help='finetune')

    # pretrain
    parser.add_argument('--max_iter', default=20000, type=float, help='maximum pretrain iteration')  # 1e6=100000
    parser.add_argument('--global_iter', default=0, type=float, help='number of iterations continue to train.')
    parser.add_argument('--save_step', default=20, type=int, help='number of iterations after which a checkpoint is saved.')
    parser.add_argument('--save_checkMin', default=1, type=int, help='number of iterations after which a checkpoint is to be compared for minimum loss.')
    # finetune
    parser.add_argument('--ft_epoch', default=50, type=int, help='maximum finetune iteration for self-expression stage')  # MvDSCN设为40
    parser.add_argument('--show-freq', default=1, type=int)
    # common
    parser.add_argument('--z_dim', default=16*8*8, type=int, help='dimension of the representation z')
    parser.add_argument('--y_dim', default=16*8*8, type=int, help='dimension of the representation y')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    # pretrain_weight
    parser.add_argument('--gamma', default=3, type=float, help='Compete - MLP')
    parser.add_argument('--alpha', default=10, type=float, help='Max - DIS')
    parser.add_argument('--mkl', default=0.05, type=float, help='MIN - mKL')
    parser.add_argument('--cmi', default=300, type=float, help='max - crossMI')  # 这里的default需要测试
    parser.add_argument('--rec', default=0.5, type=float, help='Complete - REC')
    # finetune_weight
    parser.add_argument('--c2', default=10, type=float, help='Coef 2')
    parser.add_argument('--selfExp', default=10, type=float, help='selfExpression')
    # parser.add_argument('--sim', default=100, type=float, help='similarity constraint')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='rgbd', type=str, help='dataset name: [rgbd, bbc, fashion]')
    # parser.add_argument('--image_size', default=64, type=int, help='image size. [64, 128].')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')  # 常用4,8,16
    parser.add_argument('--save_name', default='rgbd_checkpoint', type=str, help='the output directory for checkpoint.')
    parser.add_argument('--ckpt_name', default='best_min', type=str, help='load previous checkpoint. insert checkpoint filename.')

    args = parser.parse_args()

    main(args)
