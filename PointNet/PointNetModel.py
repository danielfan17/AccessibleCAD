import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

class PointNet(keras.Model):

	def __init__(self, name = None, num_classes = 10):
		super(PointNet, self).__init__(name=name)

		#### tnet ####
		bias = keras.initializers.Constant(np.eye(3).flatten())
		reg = OrthogonalRegularizer(3)

		# 1st layer set
		self.conv1 = layers.Conv1D(32, kernel_size=1, padding="valid")
		self.bn1 = layers.BatchNormalization(momentum=0.0)
		self.act1 = layers.Activation("relu")

		# 2nd layer set
		self.conv2 = layers.Conv1D(64, kernel_size=1, padding="valid")
		self.bn2 = layers.BatchNormalization(momentum=0.0)
		self.act2 = layers.Activation("relu")

		# 3rd layer set
		self.conv3 = layers.Conv1D(512, kernel_size=1, padding="valid")
		self.bn3 = layers.BatchNormalization(momentum=0.0)
		self.act3 = layers.Activation("relu")

		# 4th layer set
		self.pooling4 = layers.GlobalMaxPooling1D()
		self.dense4 = layers.Dense(256)
		self.bn4 = layers.BatchNormalization(momentum=0.0)
		self.act4 = layers.Activation("relu")

		# 5th layer set
		self.dense5 = layers.Dense(128)
		self.bn5 = layers.BatchNormalization(momentum=0.0)
		self.act5 = layers.Activation("relu")

    	# 6th layer set
		self.dense6 = layers.Dense(
			3 * 3,
			kernel_initializer="zeros",
			bias_initializer=bias,
			activity_regularizer=reg)
		self.reshape6 = layers.Reshape((3, 3))
    	# Apply affine transformation to input features
		self.dot6 = layers.Dot(axes=(2, 1)) #([inputs, feat_T])

		#### convbn ####

		# 7th layer set
		self.conv7 = layers.Conv1D(32, kernel_size=1, padding="valid")
		self.bn7 = layers.BatchNormalization(momentum=0.0)
		self.act7 = layers.Activation("relu")

		#### convbn ####

		# 8th layer set
		self.conv8 = layers.Conv1D(32, kernel_size=1, padding="valid")
		self.bn8 = layers.BatchNormalization(momentum=0.0)
		self.act8 = layers.Activation("relu")

		#### tnet ####
		bias = keras.initializers.Constant(np.eye(32).flatten())
		reg = OrthogonalRegularizer(32)

		# 9th layer set
		self.conv9 = layers.Conv1D(32, kernel_size=1, padding="valid")
		self.bn9 = layers.BatchNormalization(momentum=0.0)
		self.act9 = layers.Activation("relu")

		# 10th layer set
		self.conv10 = layers.Conv1D(64, kernel_size=1, padding="valid")
		self.bn10 = layers.BatchNormalization(momentum=0.0)
		self.act10 = layers.Activation("relu")

		# 12th layer set
		self.conv11 = layers.Conv1D(512, kernel_size=1, padding="valid")
		self.bn11 = layers.BatchNormalization(momentum=0.0)
		self.act11 = layers.Activation("relu")

		# 13th layer set
		self.pooling12 = layers.GlobalMaxPooling1D()
		self.dense12 = layers.Dense(256)
		self.bn12 = layers.BatchNormalization(momentum=0.0)
		self.act12 = layers.Activation("relu")

		# 13th layer set
		self.dense13 = layers.Dense(128)
		self.bn13= layers.BatchNormalization(momentum=0.0)
		self.act13 = layers.Activation("relu")

    	# 14th layer set
		self.dense14 = layers.Dense(
			32 * 32,
			kernel_initializer="zeros",
			bias_initializer=bias,
			activity_regularizer=reg)
		self.reshape14 = layers.Reshape((32, 32))
    	# Apply affine transformation to input features
		self.dot14 = layers.Dot(axes=(2, 1)) #([inputs, feat_T])

    	#### convbn ####

		# 15th layer set
		self.conv15 = layers.Conv1D(32, kernel_size=1, padding="valid")
		self.bn15 = layers.BatchNormalization(momentum=0.0)
		self.act15 = layers.Activation("relu")

    	#### convbn ####

		# 16th layer set
		self.conv16 = layers.Conv1D(64, kernel_size=1, padding="valid")
		self.bn16 = layers.BatchNormalization(momentum=0.0)
		self.act16 = layers.Activation("relu")

		#### convbn ####

		# 17th layer set
		self.conv17 = layers.Conv1D(512, kernel_size=1, padding="valid")
		self.bn17 = layers.BatchNormalization(momentum=0.0)
		self.act17 = layers.Activation("relu")

		#### densebn ####

		# 18th layer set
		self.pool18 = layers.GlobalMaxPooling1D()
		self.dense18 = layers.Dense(256)
		self.bn18 = layers.BatchNormalization(momentum=0.0)
		self.act18 = layers.Activation("relu")

		#### densebn ####

    	# 19th layer set
		self.drop19 = layers.Dropout(0.3)
		self.dense19 = layers.Dense(128)
		self.bn19 = layers.BatchNormalization(momentum=0.0)
		self.act19 = layers.Activation("relu")

		#### out ####

    	# 20th layer set
		self.drop20 = layers.Dropout(0.3)
		self.out20 = layers.Dense(num_classes, activation="sigmoid")

	def call(self, inputs):

		# 1st layer set
		x = self.conv1(inputs)
		x = self.bn1(x)
		x = self.act1(x)

		# 2nd layer set
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act2(x)

		# 3rd layer set
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.act3(x)

		# 4th layer set
		x = self.pooling4(x)
		x = self.dense4(x)
		x = self.bn4(x)
		x = self.act4(x)

		# 5th layer set
		x = self.dense5(x)
		x = self.bn5(x)
		x = self.act5(x)

    	# 6th layer set
		x = self.dense6(x)
		x = self.reshape6(x)
    	# Apply affine transformation to input features
		x = self.dot6([inputs, x])

		#### convbn ####

		# 7th layer set
		x = self.conv7(x)
		x = self.bn7(x)
		x = self.act7(x)

		#### convbn ####

		# 8th layer set
		x = self.conv8(x)
		x = self.bn8(x)
		x = self.act8(x)

		#### tnet ####

		# 9th layer set
		x0 = self.conv9(x)
		x = self.bn9(x0)
		x = self.act9(x)

		# 10th layer set
		x = self.conv10(x)
		x = self.bn10(x)
		x = self.act10(x)

		# 11th layer set
		x = self.conv11(x)
		x = self.bn11(x)
		x = self.act11(x)

		# 12th layer set
		x = self.pooling12(x)
		x = self.dense12(x)
		x = self.bn12(x)
		x = self.act12(x)

		# 13th layer set
		x = self.dense13(x)
		x = self.bn13(x)
		x = self.act13(x)

    	# 14th layer set
		x = self.dense14(x)
		x = self.reshape14(x)
    	# Apply affine transformation to input features
		x = self.dot14([x0, x])

    	#### convbn ####

		# 15th layer set
		x = self.conv15(x)
		x = self.bn15(x)
		x = self.act15(x)

    	#### convbn ####

		# 16th layer set
		x = self.conv16(x)
		x = self.bn16(x)
		x = self.act16(x)

		#### convbn ####

		# 17th layer set
		x = self.conv17(x)
		x = self.bn17(x)
		x = self.act17(x)

		#### densebn ####

		# 18rd layer set
		x = self.pool18(x)
		x = self.dense18(x)
		x = self.bn18(x)
		x = self.act18(x)

		#### densebn ####

    	# 19th layer set
		x = self.drop19(x)
		x = self.dense19(x)
		x = self.bn19(x)
		x = self.act19(x)

		#### out ####

    	# 20th layer set
		x = self.drop20(x)
		x = self.out20(x)

		return x

	def get_model(name = 'PointNetModel', num_classes = 10):

		return PointNet(name, num_classes)

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
