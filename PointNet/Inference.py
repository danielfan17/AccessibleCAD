import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from PointNetModel import PointNet

##### ADJUST BEFORE RUNNING #####

NUM_CLASSES = 4	# number of classes
NUM_POINTS = 2048 	# number of points to sample from mesh

MODELNAME = "ModelNet4"
CADPATH = "data/ModelNet4/chair/test/chair_0891.off"

showMesh = 0

#### 1. Load CAD to infer

# initialize test_points vector
test_points = []

# load file as mesh
mesh = trimesh.load(CADPATH)

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
model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())

# Load the state of the model
model.load_weights('models/' + MODELNAME + "_weights.index")

#### 3. Make prediction
prediction = model.predict(test_points)

print(prediction)

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