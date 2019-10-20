'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch 
import numpy as np

import torchvision
import torchvision.transforms as transforms

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class BatchMeanCrossEntropyWithLogSoftmax(nn.Module):
    def forward(self, y_hat, y):
        return -(y_hat*y).sum(dim=1).mean(dim=0)

class BatchMeanKLDivWithLogSoftmax(nn.Module):
    def forward(self, p, log_q,  log_p):
        return (p*log_p - p*log_q).sum(dim=1).mean(dim=0)


class CrossEntropyWithLogSoftmax(nn.Module):
    def forward(self, y_hat, y):
        return -(y_hat*y).mean()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def load_data(batch_size=128,is_cifar10=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if is_cifar10:
        dataset = torchvision.datasets.CIFAR10
    else:
        dataset = torchvision.datasets.CIFAR100
    trainset = dataset(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    testset = dataset(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    return trainloader, testloader

def to_one_hot(inp,num_classes):
    y_onehot = torch.cuda.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    
    return y_onehot

def get_distances(representations, sigma=1):
    rview = representations.view(representations.size(0),-1)
    distances = torch.cdist(rview,rview,p=2)
    return distances



def representations_to_adj(representations, sigma=1):
    rview = representations.view(representations.size(0),-1)
    rview =  torch.nn.functional.normalize(rview, p=2, dim=1)
    adj = torch.mm(rview,torch.t(rview))
    ind = np.diag_indices(adj.shape[0])
    adj[ind[0], ind[1]] = torch.zeros(adj.shape[0]).cuda()
    degree = torch.pow(adj.sum(dim=1),-0.5)
    degree_matrix = torch.diag(degree)
    return torch.matmul(degree_matrix,torch.matmul(adj,degree_matrix))

def train(net,trainloader,scheduler,device,optimizer,teacher=None,lambda_hkd=0,lambda_gkd=0,lambda_rkd=0,pool3_only=False,temp=4,classes=10,power=1,k=128):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = BatchMeanCrossEntropyWithLogSoftmax()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets2 = to_one_hot(targets, classes)
        targets = targets2.argmax(dim=1)
        optimizer.zero_grad()
        outputs, layers = net(inputs)
        loss = criterion(F.log_softmax(outputs,dim=-1),targets2)        
        if teacher:
            with torch.no_grad():
                teacher_output, teacher_layers = teacher(inputs)
            if lambda_hkd > 0:
                p = F.softmax(teacher_output/temp,dim=-1)
                log_q = F.log_softmax(outputs/temp,dim=-1)
                log_p = F.log_softmax(teacher_output/temp,dim=-1)
                hkd_loss = BatchMeanKLDivWithLogSoftmax()(p=p,log_q=log_q,log_p=log_p)
                loss += lambda_hkd*hkd_loss
            if lambda_rkd > 0:
                loss_rkd = 0
                zips = zip(layers,teacher_layers) if not pool3_only else zip([layers[-1]],[teacher_layers[-1]])
                for student_layer,teacher_layer in zips:
                
                    distances_teacher = get_distances(teacher_layer)
                    distances_teacher = distances_teacher[distances_teacher>0]
                    mean_teacher = distances_teacher.mean()
                    distances_teacher = distances_teacher/mean_teacher

                    distances_student = get_distances(student_layer)
                    distances_student = distances_student[distances_student>0]
                    mean_student = distances_student.mean()
                    distances_student = distances_student/mean_student
                    loss_rkd += lambda_rkd*F.smooth_l1_loss(distances_student, distances_teacher, reduction='elementwise_mean')
                if not pool3_only:
                    loss_rkd /= 3
                loss += loss_rkd
            if lambda_gkd > 0:
                loss_gkd = 0 
                zips = zip(layers,teacher_layers) if not pool3_only else zip([layers[-1]],[teacher_layers[-1]])
                for student_layer,teacher_layer in zips:
                    adj_teacher = representations_to_adj(teacher_layer)
                    adj_student = representations_to_adj(student_layer)
                    adj_teacher_p = adj_teacher
                    adj_student_p = adj_student
                    for _ in range(power-1):
                        adj_teacher_p = torch.matmul(adj_teacher_p,adj_teacher)
                        adj_student_p = torch.matmul(adj_student_p,adj_student)
                    loss_gkd += lambda_gkd*F.mse_loss(adj_teacher, adj_student, reduction='elementwise_mean')
                if not pool3_only:
                    loss_gkd /= 3
                loss += loss_gkd
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()

def test(net,testloader, device,save_name="teacher",show="accuracy"):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, layers = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if show=="accuracy":
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))                
    state = {
        'net': net,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if save_name != "no":
        torch.save(state, './checkpoint/{}.pth'.format(save_name))
    if show=="error":
        print("Test error: {:.2f}".format(100 - 100.*correct/total))
