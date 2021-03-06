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
from utils import get_concat_dataset, get_dataset, get_hue_transform, ProjectionCriterion, PredictionAccPerClass, get_lr, random_split_dataset
from pick_conf import save_confident_shifted_samples


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cuda', default=5, type=int, help='specify GPU number')
parser.add_argument('--hue_shift', default=0.5, type=float, help='in the range of [-0.5, 0.5]')
parser.add_argument('--confident_ratio', default=0.05, type=float, help='ratio to extract confident shifted samples')
parser.add_argument('--projection_weight', default=0.18, type=float, help='projection loss weight')
parser.add_argument('--projectionNetNum', default=1, type=int, help='number of rounds of projection net')
parser.add_argument('--complexNetNum', default=4, type=int, help='number of rounds of complex net')
parser.add_argument('--regularizationNum', default=1, type=int, help='number of rounds of complex net')
parser.add_argument('--epoch', default=80, type=int, help='number of training epochs')
parser.add_argument('--rand', default=0, type=int, help='random seed for dataset split (to training and validation)')
parser.add_argument('--std_projection', action='store_true', help='projection criterion including standard deviation')
# parser.add_argument('--running_average', action='store_true', help="projection criterion based on previous results")
# parser.add_argument('--with_aug', action='store_true', help='enter the second round, run without projection but with augmentation')
# parser.add_argument('--resume_final', action='store_true', help="resume the last checkpoint and run test")
args = parser.parse_args()

root_name = '%ssimple_%scomplex_%sreg_%srunavg_%sstdproj%sweight_%sepoch_%srate_%smomentum_rand%s' % (args.projectionNetNum,
                                                                                                      args.complexNetNum,
                                                                                                      args.regularizationNum,
                                                                                                      int(args.std_projection),
                                                                                                      args.projection_weight,
                                                                                                      args.epoch, args.lr,
                                                                                                      args.momentum, args.rand)


if torch.cuda.is_available():
    device = 'cuda:%s' % args.cuda
else:
    device = 'cpu'

best_validation_loss = 1e6
best_acc = 0
best_model_state = 0
start_epoch = 0

assert (args.regularizationNum <= (args.projectionNetNum + args.complexNetNum)), \
    "regularization rounds less than total number of rounds"


net_list_1 = [ProjectionNet() for _ in range(args.projectionNetNum)]
net_list_2 = [EfficientNetB0() for _ in range(args.complexNetNum)]
net_list = net_list_1 + net_list_2
del net_list_1, net_list_2

reg_list_1 = [True for _ in range(args.regularizationNum)]
reg_list_2 = [False for _ in range(len(net_list) - args.regularizationNum)]
reg_list = reg_list_1 + reg_list_2
del reg_list_1, reg_list_2

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt_concat_%s.pth' % root_name)
    round = checkpoint['round']
    print("check point in round %s" % round)
    net = net_list[round]
    net.load_state_dict(checkpoint['net'])
    net_list[round] = net
    best_validation_loss = checkpoint['validation_loss']
    start_epoch = checkpoint['epoch']
    best_model_state = checkpoint['net']

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch, trainloader, reg=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    regularizer = ProjectionCriterion()
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
            if args.std_projection:
                regularizer_loss = regularizer.compute_projection_criterion_withstd(train_outputs, test_outputs, batch_idx)
            # elif args.running_average:
            #     regularizer_loss = regularizer.compute_projection_criterion(train_outputs, test_outputs, batch_idx)
            else:
                regularizer_loss = regularizer.compute_projection_criterion_simple(train_outputs, test_outputs, batch_idx)
            loss = criterion(train_outputs, targets) + \
                   args.projection_weight * regularizer_loss

            train_loss += loss.item()
            _, predicted = train_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
            if batch_idx % 100 == 0:
                print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                      % (batch_idx, train_loss/(batch_idx + 1), 100. * correct / total, correct, total))

            loss.backward()
            optimizer.step()

    else:
        for batch_idx, (inputs, targets) in enumerate(trainloader):  # enumerate: (index, data)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc_per_class.update(predicted, targets)
            if batch_idx % 100 == 0:
                print("batch idx %s, loss %.3f , acc %.3f%% (%d/%d)"
                      % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc_per_class.output_class_prediction()


def test(epoch, round, dataset, shifted_dataset=None, test=False):          # test: when true, for final test/validation; when false, for selecting best model during training
    global best_validation_loss, best_model_state, best_acc

    net.eval()
    test_loss = 0
    accuracy_loss = 0
    distribution_mismatch_loss = 0
    correct = 0
    total = 0
    acc_per_class = PredictionAccPerClass()

    testloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False, num_workers=2)
    regularizer = ProjectionCriterion()
    if shifted_dataset is not None:     # testing regularizd model
        testloader_shifted = torch.utils.data.DataLoader(shifted_dataset, batch_size=200, shuffle=False, num_workers=2)
        with torch.no_grad():
            for batch_idx, (test_data, shifted_test_data) in enumerate(zip(testloader, testloader_shifted)):
                test_input = test_data[0]
                targets = test_data[1]
                shifted_test_input = shifted_test_data[0]
                test_input, targets, shifted_test_input = test_input.to(device), targets.to(device), shifted_test_input.to(device)

                test_outputs = net(test_input)
                shifted_test_outputs = net(shifted_test_input)

                # another criterion
                if args.std_projection:
                    regularizer_loss = regularizer.compute_projection_criterion_withstd(test_outputs, shifted_test_outputs,
                                                                                batch_idx)
                # elif args.running_average:
                #     regularizer_loss = regularizer.compute_projection_criterion(test_outputs, shifted_test_outputs,
                #                                                                 batch_idx)
                else:
                    regularizer_loss = regularizer.compute_projection_criterion_simple(test_outputs,
                                                                                       shifted_test_outputs, batch_idx)

                loss = criterion(test_outputs, targets) + args.projection_weight * regularizer_loss

                test_loss += loss.item()
                accuracy_loss += criterion(test_outputs, targets).item()
                distribution_mismatch_loss += args.projection_weight * regularizer_loss
                _, predicted = test_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc_per_class.update(predicted, targets)

            print("Accuracy + regularization loss %.3f , acc %.3f%% (%d/%d)"
                  % (test_loss/(batch_idx + 1), 100.*correct/total, correct, total))
            print("Accuracy loss %.3f"
                  % (accuracy_loss / (batch_idx + 1)))
            print("Distribution mismatch loss %.3f"
                  % (distribution_mismatch_loss / (batch_idx + 1)))
    else:                       # testing non-regularized model
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
            print("Accuracy loss %.3f , acc %.3f%% (%d/%d)"
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc_per_class.output_class_prediction()
    if not test:
        if test_loss < best_validation_loss:
            best_model_state = net.state_dict()
            print('Saving in round %s' % round)
            state = {
                'net': best_model_state,
                'validation_loss': test_loss,
                'epoch': epoch,
                'round': round,
                'learning_rate': get_lr(epoch, args.lr),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_concat_%s.pth' % root_name)
            best_validation_loss = test_loss
            return False
        else:
            print("Loss not reduced, decrease rate count: %s" % decrease_rate_count)
            return True


for aug in range(round, len(net_list)):
    print("In round %s" % aug)

    overall_trainset = get_dataset(train=True, hue=0)
    shifted_trainset = get_dataset(train=True, hue=0.5)
    random_split = np.random.randint(0, 1e6)

    training_set_len = int(overall_trainset.__len__() * 0.85)
    validation_set_len = overall_trainset.__len__() - training_set_len

    # random_state = np.random.randint(1e7)
    random_state = args.rand
    training_set, validation_set = random_split_dataset(overall_trainset,
                                                        [training_set_len, validation_set_len], random_state)
    training_set_shifted, validation_set_shifted = random_split_dataset(shifted_trainset,
                                                        [training_set_len, validation_set_len], random_state)

    if aug > 0:
        augmentation_dataset = get_concat_dataset(aug, root_name)
        print("Getting the augmentation dataset aug_%s" % root_name)
        training_set = [training_set] + augmentation_dataset
        training_set = torch.utils.data.ConcatDataset(training_set)

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=2)
    print("training_set length %s" % len(trainloader))

    if reg_list[aug]:
        trainloader_shifted = torch.utils.data.DataLoader(training_set_shifted,
                                                          batch_size=int(50000/len(trainloader)),
                                                          shuffle=True, num_workers=2)

    # Model
    print('==> Building model..')
    net = net_list[aug]
    net = net.to(device)


    end_epoch = np.max([args.epoch, start_epoch+1])
    decrease_rate_count = 0
    stop_count = 0

    for epoch in range(start_epoch, end_epoch):
        optimizer = optim.SGD(net.parameters(), lr=get_lr(epoch, args.lr), momentum=args.momentum, weight_decay=0.001) # 0.0005, momentum 0.9, lr 0.001
        if reg_list[aug]:
            train(epoch, zip(trainloader, trainloader_shifted), reg=True)
            no_improve = test(epoch, aug, validation_set, validation_set_shifted, test=False)
        else:
            train(epoch, trainloader, reg=False)
            no_improve = test(epoch, aug, validation_set, test=False)  # test False means model might be saved

        if no_improve:
            decrease_rate_count += 1
        else:
            decrease_rate_count = 0
        if decrease_rate_count >= 32:
            break

    # Question: where to extract confident samples?
    net.load_state_dict(best_model_state)
    net = net.to(device)
    save_confident_shifted_samples(net, aug, training_set_shifted, root_name, ratio=args.confident_ratio, cuda=args.cuda)

    start_epoch = 0
    best_validation_loss = 1e6
    best_acc = 0
    lr = args.lr
    decrease_rate_count = 0
    stop_count = 0

    print("Testing on complete test set:")
    net = net.to(device)
    print("On ORIGINAL dataset")
    test(100, 100, get_dataset(train=False, hue=0), test=True)

    print("On SHIFTED dataset")
    test(101, 101, get_dataset(train=False, hue=0.5), test=True)
