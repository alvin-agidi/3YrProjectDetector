import os

# import pandas as pd
import random
import sys
import urllib.request
from collections import deque

import cv2

# import datetime as dt
import numpy as np
import tensorflow as tf

# from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from tensorflow import random as rand
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model, to_categorical

from config import *

# import matplotlib.pyplot as plt

np.random.seed(SEED)
random.seed(SEED)
rand.set_seed(SEED)


def downloadPoseEstimator():
    serverPrefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
    modelZippath = tf.keras.utils.get_file(
        origin=f"{serverPrefix}/metrabs_mob3l_y4t_20211019.zip",
        extract=True,
        cache_subdir="models",
    )
    modelPath = os.path.join(os.path.dirname(modelZippath), "metrabs_mob3l_y4t")
    print(modelPath)
    return modelPath


def loadPoseEstimator():
    return tf.saved_model.load(METRABS_PATH)


def resize_frame(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    elif width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def normaliseFrame(frame):
    return frame / 255.0


def extractPoses(videoPath):
    """
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        video_poses: A list containing the poses of the video.
    """
    videoPoses = []
    videoReader = cv2.VideoCapture(videoPath)

    totalFrames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))

    skipFramesWindow = max(int(totalFrames / SEQUENCE_LENGTH), 1)

    for frameNum in range(SEQUENCE_LENGTH):
        videoReader.set(cv2.CAP_PROP_POS_FRAMES, frameNum * skipFramesWindow)
        success, frame = videoReader.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resizedFrame = resize_frame(frame, width=FRAME_HEIGHT)
        frame_tensor = tf.convert_to_tensor(resizedFrame, dtype=tf.uint8)
        print("Frame " + str(frameNum + 1) + "/" + str(SEQUENCE_LENGTH))
        # print("input shape =", tf.shape(frame_tensor))
        videoPose = poseEstimator.detect_poses(frame_tensor, skeleton=SKELETON)[
            "poses3d"
        ].numpy()
        if videoPose.shape[0] == 1:
            # print("output shape =", video_pose.shape)
            videoPoses.append(videoPose[0])
        else:
            print("Error: detected", videoPose.shape[0], "people")
            return []

    videoReader.release()
    return videoPoses


path = "https://firebasestorage.googleapis.com/v0/b/yrproject-64b5e.appspot.com/o/dhQCVLTSVZMNzDwNVDa0pumDhhm2%2Fposts%2F0.4126l9pqhdp?alt=media&token=f0d1e235-d2ce-41ca-b80e-8e683361be19"


def main(path):
    # print("0")
    # downloadPoseEstimator()
    print(extractPoses(path))


# main()
