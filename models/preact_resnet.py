'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += shortcut
        out = F.relu(out)
        return out


class PreActWideResNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=100):
        super(PreActWideResNet, self).__init__()
        self.first_block = 16
        self.in_planes = self.first_block

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, int(self.first_block*width), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(self.first_block*width*2), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(self.first_block*width*4), num_blocks[2], stride=2)

        self.linear = nn.Linear(int(self.first_block*width*4), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        pool1 = out
        out = self.layer2(pool1)
        pool2 = out
        out = self.layer3(pool2)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        pool3 = out
        out = self.linear(pool3)
        return out, [pool1,pool2,pool3]


def PreActWideResNetStart(depth=28,width=10,num_classes=100):
    n = (depth-4) //6
    return PreActWideResNet(PreActBlock, [n,n,n],width=width,num_classes=num_classes)

# test()
