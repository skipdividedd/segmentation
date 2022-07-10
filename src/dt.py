import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim
from time import time
from skimage.io import imread
from skimage.transform import resize
import os
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

images = []
lesions = []


def Dataset():
    path = os.path.abspath('../') + '\data\PH2Dataset\\'
    for root, dirs, files in os.walk(os.path.join(path, 'PH2 Dataset images')):
        if root.endswith('_Dermoscopic_Image'):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith('_lesion'):
            lesions.append(imread(os.path.join(root, files[0])))

    size = (256, 256)
    X = [resize(x, size, mode='constant', anti_aliasing=True, ) for x in images]
    Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]


    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    # Разделим наши 200 картинок на 100/50/50 для обучения, валидации и теста соответственно
    ix = np.random.choice(len(X), len(X), False)
    tr, val, ts = np.split(ix, [100, 150])

    def Dataloader(train, vl, test, batch_size=5):

        data_tr = DataLoader(list(zip(np.rollaxis(X[train], 3, 1), Y[tr, np.newaxis])),
                             batch_size=batch_size, shuffle=True)
        data_val = DataLoader(list(zip(np.rollaxis(X[vl], 3, 1), Y[val, np.newaxis])),
                              batch_size=batch_size, shuffle=True)
        data_ts = DataLoader(list(zip(np.rollaxis(X[test], 3, 1), Y[ts, np.newaxis])),
                             batch_size=batch_size, shuffle=True)

        return data_tr, data_val, data_ts

    return Dataloader(tr, val, ts)

