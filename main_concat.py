import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import pandas as pd
from models import *
from utils import get_concat_dataset, get_dataset, get_hue_transform, projection_criterion, PredictionAccPerClass
from pick_conf import save_confident_shifted_samples


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cuda', default=5, type=int, help='specify GPU number')
parser.add_argument('--hue_shift', default=0.5, type=float, help='in the range of [-0.5, 0.5]')
parser.add_argument('--confident_ratio', default=0.05, type=float, help='ratio to extract confident shifted samples')
parser.add_argument('--projection_weight', default=0.20, type=float, help='projection loss weight')
args = parser.parse_args()

if torch.cuda.is_available():
    # device = 'cuda'
    device = 'cuda:%s' % args.cuda
else:
    device = 'cpu'

best_acc_origin = 0  # best test accuracy
best_acc_shifted = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

trainset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True,
                                                transform=get_hue_transform(args.hue_shift))
trainloader_shifted = torch.utils.data.DataLoader(trainset_shifted, batch_size=100, shuffle=True, num_workers=2)
# false rate: 0.1084 for confidence ratio 5%
# moreover: confident samples are of imbalanced classes


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_concat_effinet.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc_origin = checkpoint['acc_origin']
    best_acc_shifted = checkpoint['acc_test']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch, trainloader, reg=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    acc_per_class = PredictionAccPerClass()
    if reg:
        for batch_idx, (train_data, test_data) in enumerate(trainloader):  # enumerate: (index, data)
            train_input = train_data[0]
            targets = train_data[1]
            test_input = test_data[0]
            train_input, targets, test_input = train_input.to(device), targets.to(device), test_input.to(device)

            optimizer.zero_grad()
            train_outputs = net(train_input)
            test_outputs = net(test_input)

            # another criterion
            loss = criterion(train_outputs, targets) + \
                   args.projection_weight * projection_criterion(train_outputs, test_outputs, batch_idx)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = train_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
            if batch_idx % 60 == 0:
                print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                      % (batch_idx, train_loss/(batch_idx + 1), 100. * correct / total, correct, total))
        
    else:
        for batch_idx, (inputs, targets) in enumerate(trainloader):  # enumerate: (index, data)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            # loss = criterion(outputs, targets) + criterion(outputs_concat, targets_concat)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
            if batch_idx % 60 == 0:
                print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                      % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc_per_class.output_class_prediction()


def test(epoch, round, hue=0):
    global best_acc_origin, best_acc_shifted
    best_acc = best_acc_origin if hue == 0 else best_acc_shifted

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    testloader = torch.utils.data.DataLoader(get_dataset(train=False, hue=hue),
                                                    batch_size=100, shuffle=False, num_workers=2)
    acc_per_class = PredictionAccPerClass()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
        mode = 'original' if hue == 0 else 'shifted'
        print("%s test set: loss %.3f , acc %.3f%% (%d/%d)"
              % (mode, test_loss/(batch_idx + 1), 100. * correct / total, correct, total))
        acc_per_class.output_class_prediction()

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc and hue != 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_origin': (acc if hue == 0 else best_acc_origin),
            'epoch': epoch,
            'acc_test': (acc if hue != 0 else best_acc_shifted),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_concat_projectionNet%s_%s.pth' % (round, epoch))
        if hue == 0:
            best_acc_origin = acc
        else:
            best_acc_shifted = acc


def get_lr(epoch):
    if epoch < 5:
        lr = args.lr
    elif 5 <= epoch < 20:
        lr = args.lr * 0.6
    elif 20 <= epoch < 40:
        lr = args.lr/4
    else:
        lr = args.lr/16
    return lr


# TODO: see the results by classes, see where the improvement is 
# TODO: add outer layer loop to add more confident extracted points.
# 1st function to add: get confident samples and save
# 2nd function to add: get path, create path file
# 3rd function: create new dataset and new dataloader
# TODO: try another net, more complicated.
for aug in range(0, 5):
    print("in round %s" % aug)
    # prepare augmenting dataset accoring to rounds
    overall_trainset = get_concat_dataset(aug)
    trainloader = torch.utils.data.DataLoader(overall_trainset, batch_size=128, shuffle=True, num_workers=2)
    # Model
    print('==> Building model..')
    net = ProjectionNet()
    # net = EfficientNetB0()
    net = net.to(device)

    for epoch in range(start_epoch, start_epoch+80):
        optimizer = optim.SGD(net.parameters(), lr=get_lr(epoch), momentum=0.5, weight_decay=5e-4)
        if aug == 0:
            trainloader_shifted = torch.utils.data.DataLoader(get_dataset(train=True, hue=args.hue_shift),
                                                              batch_size=128, shuffle=True, num_workers=2)
            train(epoch, zip(trainloader, trainloader_shifted), reg=True)
        else:
            train(epoch, trainloader, reg=False)
        test(epoch, aug, hue=0)
        test(epoch, aug, hue=args.hue_shift)

    # extract confident results from this round
    save_confident_shifted_samples(net, aug, get_dataset(train=True, hue=args.hue_shift), ratio=args.confident_ratio)

# Question: how to explain the normalization in transforms??
