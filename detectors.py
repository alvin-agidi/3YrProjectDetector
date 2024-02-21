import os

# import pandas as pd
import random
import urllib.request
from collections import deque

# import cv2
# import datetime as dt
import numpy as np
import tensorflow as tf

# from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from spektral.data import BatchLoader, Dataset, Graph, SingleLoader
from spektral.layers import GCNConv
from spektral.transforms import GCNFilter
from spektral.transforms.layer_preprocess import LayerPreprocess
from tensorflow import random as rand
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import TimeDistributed
from keras.layers import Flatten

from keras.optimizers import Adam

from config import *

# import matplotlib.pyplot as plt

detectors = []


class CustomLoader(SingleLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collate(self, batch):
        output = super().collate(batch)
        l1 = (
            np.reshape(np.array(output[0][0]), (SEQUENCE_LENGTH, JOINT_COUNT, 3)),
            np.reshape(
                np.array(output[0][1]), (SEQUENCE_LENGTH, JOINT_COUNT, JOINT_COUNT)
            ),
        )
        l2 = np.reshape(np.array(output[1]), (-1, SEQUENCE_LENGTH))
        return tuple((l1, l2))


class CustomDataset(Dataset):
    def __init__(self, features, labels, **kwargs):
        self.features = features.reshape((-1, JOINT_COUNT, 3))
        self.labels = labels.reshape((-1))
        super().__init__(shuffle=False, **kwargs)

    def read(self):
        return [
            Graph(x=x, a=JOINT_MATRIX, y=y) for x, y in zip(self.features, self.labels)
        ]


class GNN(Model):
    def __init__(self):
        super().__init__()
        self.graph_conv = TimeDistributed(GRAPHCONV(64, activation="tanh"))
        self.dropout = Dropout(0.8)
        # self.pool = GlobalAvgPool()
        # self.lstm = TimeDistributed(LSTM(8, activation='relu', return_sequences=True))
        self.flatten = TimeDistributed(Flatten())
        self.dense = TimeDistributed(Dense(1, activation="sigmoid"))
        self.flatten1 = Flatten()

    def call(self, inputs):
        # print(K.ndim(inputs[1]))
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        # out = self.pool(out)
        # out = self.lstm(out)
        # out = self.dropout(out)
        out = self.flatten(out)
        out = self.dense(out)
        out = self.flatten1(out)
        return out


def createLoader(features):
    dataset = CustomDataset(features)
    dataset.apply(LayerPreprocess(GRAPHCONV))
    return CustomLoader(dataset)


def loadDetectors():
    return


def predictExercises(poses):
    loader = createLoader(poses)

    res = np.array(
        [
            detector.predict(loader) / thresholds
            for detector, thresholds in zip(detectors, DETECTOR_THRESHOLDS)
        ]
    )
    # res = np.swapaxes(res, 1, 2).T

    classifications = np.array(
        [[np.argmax(values) for values in timestep] for timestep in res]
    )

    print(classifications)
