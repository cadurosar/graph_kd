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

    def __init__(self, in_planes, planes, stride=1, do_bn=True):
        super(PreActBlock, self).__init__()
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.bn2 = nn.BatchNorm2d(planes)
        self.do_bn = do_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not do_bn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not do_bn)
        self.bn1 = nn.BatchNorm2d(planes) if self.do_bn else nn.Sequential()
        self.bn2 = nn.BatchNorm2d(planes) if self.do_bn else nn.Sequential()
        self.bn3 = nn.BatchNorm2d(self.expansion*planes) if self.do_bn else nn.Sequential()
        self.alpha = nn.Parameter(torch.zeros(1))
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not do_bn),
                self.bn3
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out1 = out
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.alpha*out + shortcut
        out = F.relu(out)
        return out


class PreActWideResNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=10, do_bn=True):
        super(PreActWideResNet, self).__init__()
        self.first_block = 64
        self.in_planes = self.first_block
        self.do_bn = do_bn
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=not do_bn)
        self.bn = nn.BatchNorm2d(self.in_planes) if self.do_bn else nn.Sequential()
        self.layer1 = self._make_layer(block, int(self.first_block*width), num_blocks[0], stride=1, do_bn=do_bn)
        self.layer2 = self._make_layer(block, int(self.first_block*width*2), num_blocks[1], stride=2, do_bn=do_bn)
        self.layer3 = self._make_layer(block, int(self.first_block*width*4), num_blocks[2], stride=2, do_bn=do_bn)
        self.layer4 = self._make_layer(block, int(self.first_block*width*8), num_blocks[3], stride=2, do_bn=do_bn)
        self.linear = nn.Linear(int(self.first_block*width*8), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, do_bn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, do_bn))
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
        pool3 = out
        out = self.layer4(pool3)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        pool4 = out
        out = self.linear(pool4)
        return out, [pool1,pool2,pool3,pool4]


def PreActWideResNetStart(depth=28,width=10,num_classes=100, do_bn=True):
    n = (depth-4) //6
    return PreActWideResNet(PreActBlock, [n,n,n,n],width=width,num_classes=num_classes, do_bn=do_bn)

# test()
