'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
# from dataloader import
import os
import argparse
import numpy as np
import pandas as pd
from models import *
from utils import process_prediction


# TODO: select good model, maybe not too complicated
    # debug in pycharm, see what the prediction is, what a batch is? how does the data_loader enumerate?
    # think about how to define a new loss function

    # image is in batches, are we still going to use the statistics of all test domain data? small batches of test images are not representative of all

    # read torchvision hue transformation. Normalization needed?? ("keep the grayscale the same")

    # what is a simple projection? what is a more complicated whl learner?



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# device = 'cpu'

best_acc = 0  # best test accuracy
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

transform_hueshift = transforms.Compose([
    HueTransform(hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # TODO: hue shift here
])

trainset = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


trainset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_hueshift)
trainloader_shifted = torch.utils.data.DataLoader(trainset_shifted, batch_size=128, shuffle=True, num_workers=2)
# testset_list = [torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_hueshift) for _ in range(5)]
# projection_testset = torch.utils.data.ConcatDataset(testset_list)
# projection_testloader = torch.utils.data.DataLoader(projection_testset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_hueshift)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = EfficientNetSimple()
net = ProjectionNet()
net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/chpt_reg.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)


# Projection mismatch criterion
def projection_criterion(train_res, test_res, round):
    std_train, mean_train = torch.std_mean(train_res, dim=0)
    std_test, mean_test = torch.std_mean(test_res, dim=0)

    diff_mean = torch.dist(mean_train, mean_test)
    diff_std = torch.dist(std_train, std_test, p=1)

    if round % 30 == 0:
        print("projection criterion loss")
        print(diff_mean.cpu().data.numpy())
        print(diff_std.cpu().data.numpy())

    return 10*diff_mean + diff_std


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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
        projection_loss_weight = 0.1
        loss = criterion(train_outputs, targets) + projection_loss_weight * projection_criterion(train_outputs, test_outputs, batch_idx)
        # loss = criterion(train_outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prob, predicted = train_outputs.max(1)

        # # get test data set
        # test_batch = testloader[batch_idx % 100]
        # test_batch_np = test_batch.cpu().data.numpy()
        # np.save("test_batch%s" % batch_idx, test_batch_np)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 30 == 0:
            print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                  % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    confident_num = 5
    confident_res = np.zeros((100, confident_num))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            outputs_np = outputs.cpu().data.numpy()
            confident_index = process_prediction(outputs_np, confident_num)

            # np.save("result", outputs_np)

            predicted_truth = predicted.eq(targets).cpu().data.numpy()
            confident_res[batch_idx] = predicted_truth[confident_index]
            # print("result of confident prediction: " + str(predicted_truth[confident_index]))

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        np.save("with_reg_confident_result%s" % epoch, confident_res)
        # TODO: get confident samples and return
        print("test set: batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
              % (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/chpt_reg.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    # TODO: pass the confident samples to train_loader
