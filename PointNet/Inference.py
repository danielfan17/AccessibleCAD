import os
import glob
import trimesh
import argparse
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
CADPATHDEFAULT = "data/Test/OversizedArmchair.stl"
DATA_DIR = "data/ModelNet4"

showMesh = 1
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

    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    # for folders in class
    for i, folder in enumerate(folders):

    	# print progress
        print("processing class: {}".format(os.path.basename(folder)))

        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]

    # returns training points, training labels, and map of folder names (class) with ID
    return class_map
#

class_map = create_class_map()

#### 1. Load CAD to infer

# initialize test_points vector
test_points = []

# load file as mesh
mesh = trimesh.load(CADPATH, force = 'mesh')
mesh.merge_vertices(merge_tex=True, merge_norm=True)




# def as_mesh(scene_or_mesh):
#     """
#     Convert a possible scene to a mesh.

#     If conversion occurs, the returned mesh has only vertex and face data.
#     """
#     if isinstance(scene_or_mesh, trimesh.Scene):
#         if len(scene_or_mesh.geometry) == 0:
#             mesh = None  # empty scene
#         else:
#             # we lose texture information here
#             mesh = trimesh.util.concatenate(
#                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
#                     for g in scene_or_mesh.geometry.values()))
#     else:
#         assert(isinstance(mesh, trimesh.Trimesh))
#         mesh = scene_or_mesh
#     return mesh
# mesh = as_mesh(mesh)

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

print(class_map)
print(prediction)

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