'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import models
import models.preact_resnet
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--teacher_depth', default=28, type=int, help='')
    parser.add_argument('--teacher_width', default=1, type=float, help='')
    parser.add_argument('--depth', default=28, type=int, help='')
    parser.add_argument('--width', default=0.5, type=float, help='')
    parser.add_argument('--hkd', default=0., type=float, help='')
    parser.add_argument('--temp', default=0., type=float, help='')
    parser.add_argument('--seed', default=0., type=int, help='')
    args = parser.parse_args()
    save = "HKD_{}-{}_teaches_{}-{}_{}_{}".format(args.teacher_depth,args.teacher_width,args.depth,args.width,args.hkd,args.temp)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader = load_data(128)
    file = "checkpoint/WideResNet{}-{}_0.pth".format(args.teacher_depth,args.teacher_width)
    teacher = torch.load(file)["net"].module
    teacher = teacher.to(device)
    net = models.preact_resnet.PreActWideResNetStart(depth=args.depth,width=args.width,num_classes=10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    for epoch in range(200):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,teacher=teacher,lambda_hkd=args.hkd,temp=args.temp,classes=10)
        test(net,testloader, device, save_name=save)
if __name__ == "__main__":
    main()



