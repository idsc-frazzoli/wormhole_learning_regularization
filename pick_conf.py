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
from utils import process_prediction

parser = argparse.ArgumentParser(description='Extract confident predictions for color-shifted images')
parser.add_argument('--conf', default=5, type=int, help='confident percentage')
parser.add_argument('--cuda', default=5, type=int, help='specify GPU number')
args = parser.parse_args()

confident_num = args.conf

if torch.cuda.is_available():
    device = 'cuda'
    # device = 'cuda:%s' % args.cuda
else:
    device = 'cpu'

start_epoch = 0


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



trainset_shifted = torchvision.datasets.CIFAR10(root='../data_cifar', train=True, download=True, transform=transform_hueshift)
trainloader_shifted = torch.utils.data.DataLoader(trainset_shifted, batch_size=100, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='../data_cifar', train=False, download=True, transform=transform_hueshift)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ProjectionNet()
net = net.to(device)
print(torch.cuda.current_device())


# resume the model with best prediction on shifted test dataset
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/chpt_reg20.pth', map_location=device)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']


def predict_pick(conf):
    net.eval()
    confident_samples = []
    predict_case = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader_shifted):
            inputs = inputs.to(device)
            outputs = net(inputs)

            prob, predicted = outputs.max(1)
            outputs_np = outputs.cpu().data.numpy()
            confident_index = process_prediction(outputs_np, conf)

            predicted_truth = predicted.eq(targets.to(device)).cpu().data.numpy()
            predict_case.append(predicted_truth[confident_index])

            inputs_sampled = inputs.cpu()[confident_index].data.numpy()
            prediction_sampled = predicted.cpu()[confident_index].data.numpy()

            for id in range(conf):
                confident_samples.append((inputs_sampled[id], prediction_sampled[id]))
    np.save("reg_20_%sconf_acc" % conf, np.array(predict_case))
    return confident_samples


def transpose_image(x):
    assert(x.shape[0] == 3), "first dimension should be the channel"
    z = np.zeros((x.shape[1], x.shape[2], x.shape[0]))
    for k in range(3):
        z[:, :, k] = x[k]
    return z.astype(np.float32)


confident_samples = predict_pick(confident_num)
root = './confident_transformed_sample/'
name = 'hue_transform'
for i in range(len(confident_samples)):
    np.save(root + '%s.%d_%d' % (name, i, confident_samples[i][1]), transpose_image(confident_samples[i][0]))

