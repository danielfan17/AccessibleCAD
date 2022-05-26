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

##### ADJUST BEFORE RUNNING #####

NUM_POINTS = 2048 	# number of points to sample from mesh

MODELNAME = "ModelNet10"    # name of model to input
CADPATHDEFAULT = "data/ModelNet10/bathtub/train/bathtub_0010.off" # default model 
DATA_DIR = "data/ModelNet10" # directory of training set to get classes from   

showMesh = 0
showPointCloud = 1


#### 0. Parse script argument

# initialize parser
parser = argparse.ArgumentParser()

# parse argument
parser.add_argument('--path', type = str, default = CADPATHDEFAULT, help = 'model path(s)')

# parse argument
args = parser.parse_args()

# set cadpath based on argument
CADPATH = args.path

#### 0. Generate Class Map

#Generating class map
def create_class_map():

    class_map = {}		# list of folder names with IDs

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

    # returns training points, training labels, and map of folder names (class) with ID
    return class_map, NUM_CLASSES
#

class_map, NUM_CLASSES = create_class_map()

#### 1. Load CAD to infer

# initialize test_points vector
test_points = []

# load file as mesh
mesh = trimesh.load(CADPATH, force = 'mesh')
mesh.merge_vertices(merge_tex=True, merge_norm=True)

# option to show mesh
if showMesh == 1: 
    mesh.show()

# sample point cloud points based on mesh
mesh_sampled = mesh.sample(NUM_POINTS)

# append to test_points vector
test_points.append(mesh_sampled)

# convert to array
test_points = np.array(test_points)

#### 2. Load Model

# load model
model = PointNet.get_model(name = "PointNetModel", num_classes = NUM_CLASSES)

# compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

# build model with sample
model.train_on_batch(test_points, np.array([0]))
model.load_weights('models/' + MODELNAME + ".h5")

#### 2b. Normalize points
def normalize(points):

    norm_points = points - np.mean(points, axis = 0)
    norm_points /= np.max(np.linalg.norm(points, axis = 1))

    return norm_points

# normalize train points
for index in range(len(test_points)):

    test_points[index] = normalize(test_points[index])


#### 3. Make prediction
prediction = model.predict(test_points)[0]

#### 4. Process Results
# output loss and accuracy history
df = pd.DataFrame({'class': class_map.values(),
    'prediction': prediction})

df = df.sort_values('prediction', ascending = False)

print(df)
# print(class_map)
# print(prediction)

# show pointcloud
if showPointCloud:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(mesh_sampled[:, 0], mesh_sampled[:, 1], mesh_sampled[:,  2])
    ax.set_axis_off()
    plt.show()



# data = test_dataset.take(1)

# points, labels = list(data)[0]
# points = points[:8, ...]
# labels = labels[:8, ...]

# # run test data through model
# preds = model.predict(points)
# preds = tf.math.argmax(preds, -1)

# points = points.numpy()

# if plotresults:
#     # plot points with predicted class and label
#     fig = plt.figure(figsize=(15, 10))
#     for i in range(8):
#         ax = fig.add_subplot(2, 4, i + 1, projection="3d")
#         ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
#         ax.set_title(
#          "pred: {:}, label: {:}".format(
#                 CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
#          )
#         )
#         ax.set_axis_off()
#     plt.show()