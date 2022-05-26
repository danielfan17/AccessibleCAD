"""
Training Scropt
"""

import os
import glob
import trimesh
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from PointNetModel import PointNet
from sklearn.metrics import classification_report

##### Use command line to pass in hyperparameters #####
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("-nc", "--numclasses", help="Number of Classes", default=4, type=int)
parser.add_argument("-lr", "--learningrate", help="Learning Rate", default=0.001, type=float)
parser.add_argument("-ep", "--epoch", help="Number of Epochs", default=2, type=int)

args = parser.parse_args()
print("lr {} ep {}".format(args.learningrate, args.epoch))
#######################################################

tf.random.set_seed(1234)

##### ADJUST BEFORE RUNNING #####

# display options
showMesh = 0		# whether to show sample mesh
showPointCloud = 0  # whether to show point cloud, code is blocking

# sampling options
NUM_POINTS = 2048 	# number of points to sample from mesh
BATCH_SIZE = 32		# batch size
EPOCH_NUM = args.epoch  # number of epochs to train on
LEARNING_RATE = args.learningrate

#NUM_CLASSES = args.numclasses        # number of classes
MODELNAME = "ModelNet4"
DATA_DIR = "data/ModelNet4"

##### 2. Visualize data

mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"), force = 'mesh')

if showMesh == 1: 
    mesh.show()

# Convert to point cloud

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

    folders = glob.glob(DATA_DIR + "/*")
    folders = [f for f in folders if not os.path.isfile(f)]

    NUM_CLASSES = len(folders)

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
        NUM_CLASSES
    )

# run parsing function
train_points, test_points, train_labels, test_labels, CLASS_MAP, NUM_CLASSES = parse_dataset(
    NUM_POINTS
)   

##### 5a. Normalize datasets
print("Normalizing points for each model")

def normalize(points):

    norm_points = points - np.mean(points, axis = 0)
    norm_points /= np.max(np.linalg.norm(points, axis = 1))

    return norm_points

# normalize train points
for index in range(len(train_points)):

    train_points[index] = normalize(train_points[index])

# normalize test points
for index in range(len(test_points)):
    
    test_points[index] = normalize(test_points[index])

print("Shuffling model points")

##### 5b. Shuffle and augment training set

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

inputs = keras.Input(shape=(NUM_POINTS, 3))

model = PointNet(name = "PointNetModel", num_classes = NUM_CLASSES)

##### 7. Training Model

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["sparse_categorical_accuracy"],
)

H = model.fit(train_dataset, epochs=EPOCH_NUM, validation_data=test_dataset)

model.summary()

##### 8. Save Model

# save model weights
#model.save_weights('models/' + MODELNAME + "_weights")

model.save_weights('models/' + MODELNAME + ".h5")
#tf.keras.models.save_model(model, 'models/' + MODELNAME)

##### 9. View Results

predictions = tf.math.argmax(model.predict(test_points, batch_size=BATCH_SIZE), -1)
report = classification_report(test_labels, predictions, target_names= list(CLASS_MAP.values()), output_dict = True)
print(report)
##### 10. Output result

# output report to csv
df = pd.DataFrame(report).transpose()
df.to_csv("results/" + MODELNAME + "_classAccuracy.csv")

# determine the number of epochs
N = np.arange(0, EPOCH_NUM)
loss = H.history["loss"]
val_loss = H.history["val_loss"]
accuracy = H.history["sparse_categorical_accuracy"]
val_accuracy = H.history["val_sparse_categorical_accuracy"]

# output loss and accuracy history
df = pd.DataFrame({'epoch': N,
    'loss': loss,
    'val_loss': val_loss,
    'accuracy': accuracy,
    'val_accuracy': val_accuracy})
df.to_csv("results/" + MODELNAME + "_history.csv")

plt.rcParams["figure.figsize"] = (10,4)

plt.subplot(1, 2, 1)
title = "Training Loss"
plt.plot(N, loss, label="train_loss")
plt.plot(N, val_loss, label="val_loss")
plt.title(title)
plt.xlabel("Epoch #")
plt.legend()

plt.subplot(1, 2, 2)
title = "Training Accuracy"
plt.plot(N, accuracy, label="train_accuracy")
plt.plot(N, val_accuracy, label="val_accuracy")
plt.title(title)
plt.xlabel("Epoch #")
plt.legend()

plt.savefig('results/' + MODELNAME + ".png")
plt.show()
#plt.savefig(args["plot"])

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
