'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


add_shortcut_to_outputs = False

class ResnetBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, do_bn=True):
        super(ResnetBlock, self).__init__()
        self.do_bn = do_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not do_bn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not do_bn)
        self.bn1 = nn.BatchNorm2d(planes) if self.do_bn else nn.Sequential()
        self.bn2 = nn.BatchNorm2d(planes) if self.do_bn else nn.Sequential()
        self.bn3 = nn.BatchNorm2d(self.expansion*planes) if self.do_bn else nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not do_bn),
                self.bn3
            )

    def forward(self, x):
        outputs = list()
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        outputs.append(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        if add_shortcut_to_outputs and hasattr(self, 'shortcut'):
            outputs.append(shortcut)
        out = out + shortcut
        out = F.relu(out)
        outputs.append(out)
        return out, outputs


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=10, do_bn=True):
        super(ResNet, self).__init__()
        self.first_block = 64
        self.in_planes = self.first_block
        in_planes = self.in_planes
        self.do_bn = do_bn
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=not do_bn)
        self.bn = nn.BatchNorm2d(self.in_planes) if self.do_bn else nn.Sequential()
        self.blocks = nn.ModuleList()
        for idx, num_block in enumerate(num_blocks):
            if idx == 0:
                stride = 1
                multiply = False
            else:
                stride = 2
                multiply = True
            for _ in range(num_block):
                out_planes = in_planes if not multiply else in_planes*2
                self.blocks.append(block(in_planes,out_planes,stride,do_bn))
                in_planes = out_planes
                stride = 1
                multiply = False                
        self.linear = nn.Linear(out_planes, num_classes)

    def forward(self, x):
        outputs = list()
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        outputs.append(out)
        for idx, block in enumerate(self.blocks):
            out, outs = block(out)
            if idx < len(self.blocks)-1:
                outputs.extend(outs)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        outputs.append(out)
        out = self.linear(out)
        return out, outputs


def ResNetStart(depth=28,width=10,num_classes=100, do_bn=True):
    n = (depth-4) //6
    return ResNet(ResnetBlock, [n,n,n,n],width=width,num_classes=num_classes, do_bn=do_bn)

