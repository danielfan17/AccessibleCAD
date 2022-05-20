"""
Training Scropt
"""

import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)

###### User-defined Options #######

# display options
showMesh = 0		# whether to show sample mesh
showPointCloud = 0  # whether to show point cloud, code is blocking

# sampling options
NUM_POINTS = 2048 	# number of points to sample from mesh
NUM_CLASSES = 10	# number of classes
BATCH_SIZE = 32		# batch size
EPOCH_NUM = 2		# number of epochs to train on

MODELNAME = "ModelNet4"

##### 1. Download data

DATA_DIR = "data/ModelNet4"

##### 2. Visualize data

mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))

if showMesh == 1: 
    mesh.show()

##### 3. Convert to point cloud

points = mesh.sample(2048)

if showPointCloud:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()

##### 4. Parse data of training set

def parse_dataset(num_points=2048):

    train_points = []	# list of training points
    train_labels = []	# list of training labels
    test_points = []
    test_labels = []
    class_map = {}		# list of folder names with IDs

    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    # for folders in class
    for i, folder in enumerate(folders):

    	# print progress
        print("processing class: {}".format(os.path.basename(folder)))

        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]

        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        # add training data to points and labels
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    # returns training points, training labels, and map of folder names (class) with ID
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

# run parsing function
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

##### 5. Shuffle and augment training set

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

# get points and labels as slices, shuffle, and augment
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

##### 6. Defining model

# Each convolution and fully-connected layer (with exception for end layers) consists of
# Convolution / Dense -> Batch Normalization -> ReLU Activation.
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

# def OrthogonalRegularizer2(num_features, l2reg = 0.001):
# 	eye = tf.eye(num_features)
# 	x = tf.reshape(x, (-1, num_features, num_features))
# 	xxt = tf.tensordot(x, x, axes=(2, 2))
# 	xxt = tf.reshape(xxt, (-1, num_features, num_features))
# 	return tf.reduce_sum(l2reg * tf.square(xxt - eye))
"""
The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    #reg = OrthogonalRegularizer(num_features)
    reg = regularizers.OrthogonalRegularizer(factor=0.001, mode="columns")

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

"""
Network architecture published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.
"""

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

##### 7. Training Model

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=EPOCH_NUM, validation_data=test_dataset)

##### 8. Save Model

# serialize model to JSON
model.save('test')