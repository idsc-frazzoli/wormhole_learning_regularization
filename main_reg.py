'''Train CIFAR10 with PyTorch.'''
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
from pick_conf import process_prediction
from utils import PredictionAccPerClass, get_lr


# TODO: select good model, maybe not too complicated
    # debug in pycharm, see what the prediction is, what a batch is? how does the data_loader enumerate?
    # think about how to define a new loss function

    # image is in batches, are we still going to use the statistics of all test domain data? small batches of test images are not representative of all

    # read torchvision hue transformation. Normalization needed?? ("keep the grayscale the same")

    # what is a simple projection? what is a more complicated whl learner?


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cuda', default=5, type=int, help='specify GPU number')
parser.add_argument('--projection_weight', default=0.20, type=float, help='projection loss weight')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda:%s' % args.cuda
else:
    device = 'cpu'
# device = 'cpu'

best_acc_original = 0  # best test accuracy
best_acc_shifted = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
projection_loss_weight = args.projection_weight


# def hue_transform(image, hue):
class HueTransform:
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, x):
        return transforms.functional.adjust_hue(x, self.hue)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_hueshift = transforms.Compose([
    HueTransform(hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

trainset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_hueshift)
trainloader_shifted = torch.utils.data.DataLoader(trainset_shifted, batch_size=128, shuffle=True, num_workers=2)

testset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_hueshift)
testloader_shifted = torch.utils.data.DataLoader(testset_shifted, batch_size=100, shuffle=False, num_workers=2)

testset_original = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_train)
testloader_original = torch.utils.data.DataLoader(testset_original, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ProjectionNet()
# net = EfficientNetB0()
net = net.to(device)
print(torch.cuda.current_device())


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_reg%8.2f.pth' % projection_loss_weight)
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    best_acc_original = checkpoint['origin_acc']
    best_acc_shifted = checkpoint['shifted_acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)


# Projection mismatch criterion
def projection_criterion(train_res, test_res, round):
    std_train, mean_train = torch.std_mean(train_res, dim=0)
    std_test, mean_test = torch.std_mean(test_res, dim=0)

    diff_mean = torch.dist(mean_train, mean_test)
    diff_std = torch.dist(std_train, std_test, p=1)

    if round % 60 == 0:
        print("projection criterion loss")
        print(diff_mean.cpu().data.numpy())
        print(diff_std.cpu().data.numpy())

    return 10*diff_mean + diff_std


# Training
def train(epoch):
    global projection_loss_weight
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    acc_per_class = PredictionAccPerClass()
    # for batch_idx, (inputs, targets) in enumerate(trainloader):  # enumerate: (index, data)
    for batch_idx, (train_data, test_data) in enumerate(zip(trainloader, trainloader_shifted)):  # enumerate: (index, data)
        train_input = train_data[0]
        targets = train_data[1]
        test_input = test_data[0]
        train_input, targets, test_input = train_input.to(device), targets.to(device), test_input.to(device)

        optimizer.zero_grad()
        train_outputs = net(train_input)
        test_outputs = net(test_input)

        # another criterion
        loss = criterion(train_outputs, targets) + projection_loss_weight * projection_criterion(train_outputs, test_outputs, batch_idx)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prob, predicted = train_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc_per_class.update(predicted, targets)

        if batch_idx % 60 == 0:
            print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                  % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc_per_class.output_class_prediction()


def test(epoch, shift=False):
    global best_acc_original, best_acc_shifted
    best_acc = best_acc_shifted if shift else best_acc_original
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    testloader = testloader_shifted if shift else testloader_original
    acc_per_class = PredictionAccPerClass()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
        print("shift %s test set: batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
              % (shift, batch_idx, test_loss/(batch_idx + 1), 100.*correct/total, correct, total))
        acc_per_class.output_class_prediction()

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'origin_acc': acc if not shift else best_acc_original,
            'shifted_acc': acc if shift else best_acc_shifted,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_reg%8.3f.pth' % projection_loss_weight)
        if shift:
            best_acc_shifted = acc
        else:
            best_acc_original = acc


for epoch in range(start_epoch, start_epoch+200):
    optimizer = optim.SGD(net.parameters(), lr=get_lr(epoch, args.lr), momentum=0.5, weight_decay=5e-4)
    train(epoch)
    test(epoch, shift=False)
    test(epoch, shift=True)
