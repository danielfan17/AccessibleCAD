import os
import glob
import trimesh
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from PointNetModel import PointNet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


##### ADJUST BEFORE RUNNING #####

NUM_POINTS = 2048   # number of points to sample from mesh

MODELNAME = "ModelNet10"    # name of model to input
CADPATHDEFAULT = "data/ModelNet40/airplane/train/airplane_0002.off" # default model 
DATA_DIR = "data/ModelNet10" # directory of training set to get classes from   

BATCH_SIZE = 32

#### 0. Parse script argument

# initialize parser
parser = argparse.ArgumentParser()

# parse argument
parser.add_argument('--path', type = str, default = CADPATHDEFAULT, help = 'model path(s)')

# parse argument
args = parser.parse_args()

# set cadpath based on argument
CADPATH = args.path

##### 1. Parse data of training set

def parse_dataset(num_points=2048):

    test_points = []
    test_labels = []
    class_map = {}      # list of folder names with IDs

    folders = glob.glob(DATA_DIR + "/*")
    folders = [f for f in folders if not os.path.isfile(f)]
    folders = sorted(folders)

    NUM_CLASSES = len(folders)

    # for folders in class
    for i, folder in enumerate(folders):

        # print progress
        print("processing class: {}".format(os.path.basename(folder)))

        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]

        # gather all files
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    # returns training points, training labels, and map of folder names (class) with ID
    return (
        np.array(test_points),
        np.array(test_labels),
        class_map,
        NUM_CLASSES
    )

# run parsing function
test_points, test_labels, CLASS_MAP, NUM_CLASSES = parse_dataset(
    NUM_POINTS
)   

##### 2. Normalize datasets
print("Normalizing points for each model")

def normalize(points):

    norm_points = points - np.mean(points, axis = 0)
    norm_points /= np.max(np.linalg.norm(points, axis = 1))

    return norm_points

# normalize test points
for index in range(len(test_points)):
    
    test_points[index] = normalize(test_points[index])

print("Shuffling model points")

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

#### 3. Load Model

# load model
model = PointNet.get_model(name = "PointNetModel", num_classes = NUM_CLASSES)

# compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

# build model with sample
X = np.array([test_points[0].tolist()])
H = model.train_on_batch(X, np.array([0]))
model.load_weights('models/' + MODELNAME + ".h5")

##### 4. View validation results and confusion matrix

predictions = tf.math.argmax(model.predict(test_points, batch_size=BATCH_SIZE), -1)
report = classification_report(test_labels, predictions, target_names= list(CLASS_MAP.values()), output_dict = True)
print(report)

matrix = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
    display_labels=CLASS_MAP.values()).plot()
plt.show()


