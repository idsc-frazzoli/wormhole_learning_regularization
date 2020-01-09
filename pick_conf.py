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
from utils import PredictionAccPerClass


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def process_prediction(res, i):  # get indexes of confident samples
    sum_class = np.sum(np.absolute(res), axis=1)
    pred_prob = np.max(res, axis=1)
    ratio = np.divide(pred_prob, sum_class)
    ratio_pd = pd.DataFrame(data=ratio, columns=['max_pred_ratio'])
    sorted_pd = ratio_pd.sort_values('max_pred_ratio', ascending=False)
    return list(sorted_pd[0:i].index)


def predict_pick(model, conf, dataloader, cuda=5):
    net = model
    net.eval()
    confident_samples = []
    predict_case = []
    acc_per_class = PredictionAccPerClass()
    if torch.cuda.is_available():
        # device = 'cuda'
        device = 'cuda:%s' % cuda
    else:
        device = 'cpu'
    net = net.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = net(inputs)

            prob, predicted = outputs.max(1)
            outputs_np = outputs.cpu().data.numpy()
            confident_index = process_prediction(outputs_np, conf)

            predicted_truth = predicted.eq(targets.to(device)).cpu().data.numpy()
            predict_case.append(predicted_truth[confident_index])

            prediction_sampled_device = predicted[confident_index]
            targets_sampled_device = targets[confident_index]

            inputs_sampled = inputs.cpu()[confident_index].data.numpy()
            prediction_sampled = prediction_sampled_device.cpu().data.numpy()

            acc_per_class.update(prediction_sampled_device, targets_sampled_device)

            for id in range(conf):
                confident_samples.append((inputs_sampled[id], prediction_sampled[id]))
    predict_result = np.array(predict_case).astype(int).astype(float)
    print("overall accuracy of confident samples: %8.3f" % np.average(predict_result))
    acc_per_class.output_class_prediction()

    return confident_samples


def transpose_image(x):
    assert(x.shape[0] == 3), "first dimension should be the channel"
    z = np.zeros((x.shape[1], x.shape[2], x.shape[0]))
    for k in range(3):
        z[:, :, k] = x[k]
    return z.astype(np.float32)


def save_confident_shifted_samples(model, round, dataset, dir_name, ratio=0.05):
    batch_size = 100
    trainloader_shifted = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    confident_samples = predict_pick(model=model, conf=int(np.rint(ratio*batch_size)), dataloader=trainloader_shifted)

    root = './confident_samples/%s/confident_sample%s' % (dir_name, round)
    if os.path.exists(root):
        import glob
        files = glob.glob(root + '/*')
        for f in files:
            os.remove(f)
        print("Previous saved samples removed.")
    else:
        os.makedirs(root)
        print("New directory for samples made.")

    name = 'hue_transform'
    for sample in range(len(confident_samples)):
        np.save(root + '/%s.%d_%d' % (name, sample, confident_samples[sample][1]),
                transpose_image(confident_samples[sample][0]))

    path_train = []
    files = os.listdir(root)
    for file in files:
        file_dir = os.path.join(root, file)
        path_train.append(file_dir)
    # random.shuffle(path_train)
    with open('./confident_samples/%s/path_shifted_train%s.txt' % (dir_name, round), 'w') as f:
        for j in range(len(path_train)):
            path = path_train[j]
            if j != len(path_train) - 1:
                path = path + ' '
            f.write(path)


