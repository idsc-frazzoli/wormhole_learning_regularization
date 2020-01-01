import pandas as pd
import numpy as np
# from sklearn.metrics import balanced_accuracy_score


def process_acc(res):
    acc = []
    for i in range(1, 6):
        cut = res[:, 0:i].reshape(-1, )
        true_n = np.where(cut == 1)[0].shape[0]
        total_n = cut.shape[0]
        acc.append(true_n/total_n)
    return acc


for epoch in range(200):
    res = np.load("./confident_accuracy_result/with_reg20_confident_result%s.npy" % epoch)
    print("epoch: %s" %epoch)
    print(process_acc(res))