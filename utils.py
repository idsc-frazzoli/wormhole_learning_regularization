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
import torch.nn.init as init
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch
from sample_new_images import Signal
from torch._utils import _accumulate
from torch.utils.data import Dataset, Subset

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


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

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


class PredictionAccPerClass:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.class_correct = list(0. for i in range(10))
        self.class_total = list(0. for i in range(10))

    def update(self, prediction, targets):
        preds = prediction.cpu().data.numpy()
        targ = targets.cpu().data.numpy()
        c = (preds == targ).squeeze()
        for i in range(preds.shape[0]):
            label = targ[i]
            self.class_correct[label] += c[i].item()
            self.class_total[label] += 1

    def output_class_prediction(self):
        for i in range(10):
            if self.class_total[i] == 0:
                acc = 0
            else:
                acc = 100 * self.class_correct[i] / self.class_total[i]
            print('Accuracy of %5s : %2d %%, %4d / %5d' % (
                self.classes[i], acc,
                self.class_correct[i], self.class_total[i]))


class ProjectionCriterion():
    def __init__(self):
        self.source_mean = 0
        self.target_mean = 0
        self.init = True
        self.alpha = 0.5

    def compute_projection_criterion(self, train_res, test_res, round):
        mean_train = torch.mean(train_res, dim=0)
        mean_test = torch.mean(test_res, dim=0)
        if self.init:
            self.source_mean = mean_train
            self.target_mean = mean_test
            self.init = False
        else:
            self.source_mean = self.alpha * mean_train + (1 - self.alpha) * self.source_mean.detach()
            self.target_mean = self.alpha * mean_test + (1 - self.alpha) * self.target_mean.detach()

        if round % 100 == 0:
            print("projection criterion loss %8.5f" % (torch.dist(self.source_mean, self.target_mean).cpu().data.numpy()))
        return torch.dist(self.source_mean, self.target_mean)

    def compute_projection_criterion_simple(self, train_res, test_res, round):
        mean_train = torch.mean(train_res, dim=0)
        mean_test = torch.mean(test_res, dim=0)
        dist = torch.dist(mean_train, mean_test)
        if round % 100 == 0:
            print(
                "projection criterion loss %8.5f" % (dist.cpu().data.numpy()))
        return dist

    def compute_projection_criterion_withstd(self, train_res, test_res, round):
        mean_train = torch.mean(train_res, dim=0)
        mean_test = torch.mean(test_res, dim=0)
        if self.init:
            self.source_mean = mean_train
            self.target_mean = mean_test
            self.init = False
        else:
            self.source_mean = self.alpha * mean_train + (1 - self.alpha) * self.source_mean.detach()
            self.target_mean = self.alpha * mean_test + (1 - self.alpha) * self.target_mean.detach()

        std_train = torch.std(train_res, dim=0)
        std_test = torch.std(test_res, dim=0)

        diff_mean = torch.dist(self.source_mean, self.target_mean)
        diff_std = torch.dist(std_train, std_test)  # on github I wrote p=1

        if round % 100 == 0:
            print("projection loss acc %8.5f, std %8.5f" %
                  (diff_mean.item(), diff_std.item()))
        assert (not np.isnan(diff_mean.cpu().data.numpy()))
        assert (not np.isnan(diff_std.cpu().data.numpy()))
        return diff_mean + 0.1 * diff_std

# def projection_criterion(train_res, test_res, round):
#     std_train, mean_train = torch.std_mean(train_res, dim=0)
#     std_test, mean_test = torch.std_mean(test_res, dim=0)
#
#     diff_mean = torch.dist(mean_train, mean_test)
#     diff_std = torch.dist(std_train, std_test, p=1)
#
#     if round % 100 == 0:
#         print("projection criterion loss %8.5f, %8.5f" %
#               (diff_mean.cpu().data.numpy(), diff_std.cpu().data.numpy()))
#     assert (not np.isnan(diff_mean.cpu().data.numpy()))
#     assert (not np.isnan(diff_std.cpu().data.numpy()))
#     return 10*diff_mean + diff_std


def get_lr(epoch, rate):
    if epoch < 10:
        lr = rate
    elif 10 <= epoch < 40:
        lr = rate/4
    else:
        lr = rate/20
    return lr


# def hue_transform(image, hue):
class HueTransform:
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, x):
        return transforms.functional.adjust_hue(x, self.hue)


def get_hue_transform(hue=0.5):
    if hue != 0:
        transform_hueshift = transforms.Compose([
            HueTransform(hue=hue),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_hueshift = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_hueshift


def get_concat_dataset(round, dir_name):
    transform_concat = transforms.Compose([
        transforms.ToTensor(),
    ])

    assert(round > 0), "No augmentation needed in the initial round"
    print("get %s concatenated dataset" % round)
    concat_set = []
    for i in range(round):
        # get the i-th confident sample sets
        trainset_concat = Signal('./confident_samples/%s/path_shifted_train%s.txt' % (dir_name, i),
                                 transform=transform_concat)
        concat_set.append(trainset_concat)
    return concat_set


def get_dataset(train=True, hue=0.5):
    assert(-0.5 <= hue <= 0.5), "hue shift range: [-0.5, 0.5]"
    if hue == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            HueTransform(hue=hue),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return torchvision.datasets.CIFAR10(root='../data_cifar', train=train, download=True,
                                                  transform=transform)


def random_split_dataset(dataset, lengths, seed=0):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    r = np.random.RandomState(1234)
    r.seed(seed)
    indices = r.permutation(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def kfold_split(dataset, fold=5, seed=0):
    total_len = dataset.__len__()
    lengths = []
    for _ in range(fold-1):
        lengths.append(int(total_len / fold))
    lengths.append(total_len - (fold-1) * int(total_len/fold))

    r = np.random.RandomState(1234)
    r.seed(seed)

    indices = r.permutation(total_len).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

