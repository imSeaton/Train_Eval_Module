import sys
from dataset import get_dataset
from train_eval import cross_validation_with_val_set
from param_parser import parameter_parser
import torch
import numpy as np
import random
from helper import set_gpu
from utils import logger


if __name__ == '__main__':
    # get the args
    args = parameter_parser()
    print(f"args: {args}")
    device = set_gpu(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    dataset = get_dataset(name=args.dataset, root="../data/"+args.dataset)
    cross_validation_with_val_set(
        args,
        dataset,
        device,
        logger=logger
    )