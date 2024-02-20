CLASS_LIST = ['barbell_biceps_curl', 'deadlift', 'lat_pulldown', 'lateral_raise']
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
SKELETON = 'smpl_24'
METRABS_PATH = ""
DETECTOR_PATHS = []
DETECTOR_THRESHOLDS = []
# JOINT_NAMES = model.per_skeleton_joint_names[SKELETON].numpy().astype(str)
# JOINT_EDGES = model.per_skeleton_joint_edges[SKELETON].numpy()
JOINT_NAMES = ['pelv', 'lhip', 'rhip', 'spi1', 'lkne', 'rkne', 'spi2', 'lank', 'rank', 'spi3', 'ltoe', 'rtoe', 'neck', 'lcla', 'rcla', 'head', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri', 'lhan', 'rhan']
JOINT_EDGES = [[1, 4], [1, 0], [2, 5], [2, 0], [3, 6], [3, 0], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [12, 13], [12, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
JOINT_COUNT = len(JOINT_NAMES)
# JOINT_EDGES_DF = pd.DataFrame(
#     {"source": JOINT_EDGES_TEXT[:,0], "target": JOINT_EDGES_TEXT[:,1]}
# )
JOINT_MATRIX = np.zeros([JOINT_COUNT, JOINT_COUNT], np.float32)
for i,j in JOINT_EDGES:
    JOINT_MATRIX[i, j] = JOINT_MATRIX[j, i] = 1