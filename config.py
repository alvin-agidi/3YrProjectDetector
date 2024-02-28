import numpy as np
from spektral.layers import GCNConv
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
from keras.layers import TimeDistributed, Flatten

from keras.optimizers import Adam

CLASS_LIST = ["barbell_biceps_curl", "deadlift", "lat_pulldown", "lateral_raise"]
# Specify the height and width to which each video frame will be resized in our dataset.
FRAME_HEIGHT, FRAME_WIDTH = 512, 512
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 16
TEST_SPLIT = 0.25
VALIDATION_SPLIT = 0.2
EPOCH_COUNT = 50
BATCH_SIZE = 1
PATIENCE = 10
JOINT_COUNT = 24
USE_UCF101 = False
SEED = 0
SKELETON = "smpl_24"
METRABS_PATH = "metrabs"
DETECTOR_PATHS = []
DETECTOR_THRESHOLDS = []
# JOINT_NAMES = model.per_skeleton_joint_names[SKELETON].numpy().astype(str)
# JOINT_EDGES = model.per_skeleton_joint_edges[SKELETON].numpy()
JOINT_NAMES = [
    "pelv",
    "lhip",
    "rhip",
    "spi1",
    "lkne",
    "rkne",
    "spi2",
    "lank",
    "rank",
    "spi3",
    "ltoe",
    "rtoe",
    "neck",
    "lcla",
    "rcla",
    "head",
    "lsho",
    "rsho",
    "lelb",
    "relb",
    "lwri",
    "rwri",
    "lhan",
    "rhan",
]
JOINT_EDGES = [
    [1, 4],
    [1, 0],
    [2, 5],
    [2, 0],
    [3, 6],
    [3, 0],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [12, 13],
    [12, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 22],
    [21, 23],
]
JOINT_COUNT = len(JOINT_NAMES)
# JOINT_EDGES_DF = pd.DataFrame(
#     {"source": JOINT_EDGES_TEXT[:,0], "target": JOINT_EDGES_TEXT[:,1]}
# )
JOINT_MATRIX = np.zeros([JOINT_COUNT, JOINT_COUNT], np.float32)
for i, j in JOINT_EDGES:
    JOINT_MATRIX[i, j] = JOINT_MATRIX[j, i] = 1

GRAPHCONV = GCNConv


class CustomLoader(BatchLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(
            shuffle=False, batch_size=SEQUENCE_LENGTH * BATCH_SIZE, *args, **kwargs
        )

    def collate(self, batch):
        output = super().collate(batch)
        l1 = (
            np.reshape(np.array(output[0][0]), (-1, SEQUENCE_LENGTH, JOINT_COUNT, 3)),
            np.reshape(
                np.array(output[0][1]), (-1, SEQUENCE_LENGTH, JOINT_COUNT, JOINT_COUNT)
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
