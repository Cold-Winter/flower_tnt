import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
 
from dataset import FlowerDataSet
# from models import TSN
from transforms import *
from opts import parser
import scipy.io as sio


def main():
    args = parser.parse_args()
    splitanno = sio.loadmat('./dataanno/anno.mat')
#     labelmap = sio.loadmat('./dataanno/setid.mat')
    trainid = splitanno['trnid'][0].tolist()
    valid = splitanno['valid'][0].tolist()
    testid = splitanno['tstid'][0].tolist()
    labellist = splitanno['labels'][0].tolist()
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = FlowerDataSet(indexlist = trainid, labellist = labellist,
                        transform=torchvision.transforms.Compose([
                       torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)]),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for i, (input, target) in enumerate(train_loader):
        print input.size()


if __name__ == '__main__':
    main()       
        