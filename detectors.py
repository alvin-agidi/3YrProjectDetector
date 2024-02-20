from spektral.data import Dataset, Graph, BatchLoader, SingleLoader
from spektral.transforms import GCNFilter
from spektral.transforms.layer_preprocess import LayerPreprocess
from spektral.layers import GCNConv
# from scipy import sparse
from tensorflow import random as rand
import os
# import cv2
import datetime as dt
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from collections import deque
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from config import *

class CustomLoader(SingleLoader):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def collate(self, batch):
        output = super().collate(batch)
        l1 = (
            np.reshape(np.array(output[0][0]), (SEQUENCE_LENGTH, JOINT_COUNT, 3)),
            np.reshape(np.array(output[0][1]), (SEQUENCE_LENGTH, JOINT_COUNT, JOINT_COUNT))
        )
        l2 = np.reshape(np.array(output[1]), (-1, SEQUENCE_LENGTH))
        return tuple((l1,l2))

class CustomDataset(Dataset):
    def __init__(self, features, labels, **kwargs):
        self.features = features.reshape((-1, JOINT_COUNT, 3))
        self.labels = labels.reshape((-1))
        super().__init__(shuffle=False, **kwargs)

    def read(self):
        return [Graph(x=x, a=JOINT_MATRIX, y=y) for x, y in zip(self.features, self.labels)]

def predictExercisees():
    res = np.array([detector.predict(loader)/thresholds for detector, thresholds in zip(detectors, detector_thresholds)])
    # res = np.swapaxes(res, 1, 2).T

    classifications = np.array([[np.argmax(values) for values in timestep] for timestep in res])

    print(classifications)