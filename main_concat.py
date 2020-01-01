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
from sample_new_images import Signal
from utils import process_prediction


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cuda', default=5, type=int, help='specify GPU number')
args = parser.parse_args()

if torch.cuda.is_available():
    # device = 'cuda'
    device = 'cuda:%s' % args.cuda
else:
    device = 'cpu'

best_acc_origin = 0  # best test accuracy
best_acc_shifted = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


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

transform_concat = transforms.Compose([
    transforms.ToTensor(),
])

transform_hueshift = transforms.Compose([
    HueTransform(hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_train)
# false rate: 0.1084 for confidence ratio 5%
# moreover: confident samples are of imbalanced classes
trainset_concat = Signal('path_shifted_train.txt', transform=transform_concat, train=True, test=False)
overall_trainset = torch.utils.data.ConcatDataset([trainset, trainset_concat])
trainloader = torch.utils.data.DataLoader(overall_trainset, batch_size=128, shuffle=True, num_workers=2)

testset_origin = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_train)
testloader_origin = torch.utils.data.DataLoader(testset_origin, batch_size=100, shuffle=False, num_workers=2)

testset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_hueshift)
testloader_shifted = torch.utils.data.DataLoader(testset_shifted, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = ProjectionNet()
net = EfficientNetB0()
net = net.to(device)
print(torch.cuda.current_device())

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
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.5, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):  # enumerate: (index, data)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        # outputs_concat = net(inputs_concat)

        # loss = criterion(outputs, targets) + criterion(outputs_concat, targets_concat)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prob, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 30 == 0:
            print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                  % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test_origin(epoch):
    global best_acc_origin
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_origin):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # TODO: get confident samples and return
        print("original test set: batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
              % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_origin:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_origin': acc,
            'epoch': epoch,
            'acc_test': best_acc_shifted
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_concat_effinet.pth')
        best_acc_origin = acc


def test_shifted(epoch):
    global best_acc_shifted
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_shifted):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # TODO: get confident samples and return
        print("shifted test set: batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
              % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc_shifted:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc_origin': best_acc_origin,
            'epoch': epoch,
            'acc_test': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_concat_effinet.pth')
        best_acc_shifted = acc


def get_lr(epoch):
    if epoch < 50:
        lr = args.lr
    elif epoch >= 50 and epoch < 100:
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
for epoch in range(start_epoch, start_epoch+200):  ## maybe 70 - 100 is enough
    optimizer = optim.SGD(net.parameters(), lr=get_lr(epoch), momentum=0.5, weight_decay=5e-4)
    train(epoch)
    test_origin(epoch)
    test_shifted(epoch)
