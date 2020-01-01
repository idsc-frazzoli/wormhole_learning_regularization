import os
import random
from collections import Counter

root_dir = './confident_transformed_sample'
path_train = []
files = os.listdir(root_dir)
for file in files:
    file_dir = os.path.join(root_dir, file)
    path_train.append(file_dir)

random.shuffle(path_train)

with open('./path_shifted_train.txt', 'w') as f:
    for j in range(len(path_train)):
        path = path_train[j]
        if j != len(path_train)-1:
            path = path + ' '
        f.write(path)

# types = []
# for path in path_train:
#     a = path.split("_")[4].split(".")[0]
#     types.append(int(a))
#
# b = Counter(types)
#
# print(b)