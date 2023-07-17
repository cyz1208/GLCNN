import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, \
	Flatten, Dense, Activation, BatchNormalization, Dropout, concatenate, Add


def wide_deep(Input_1, Input_2):
	x_1 = Lenet5(Input_1)
	Output = concatenate([x_1, Input_2])
	Output = Dense(2000)(Output)
	Output = Activation(tf.nn.relu)(Output)
	Output = Dropout(0.2)(Output)
	Output = Dense(200)(Output)
	Output = Activation(tf.nn.relu)(Output)
	Output = Dropout(0.2)(Output)
	Output = Dense(1)(Output)
	return Output


def Lenet5(Input):
	# Input shape (32,32)
	x = Conv2D(6, 5, 1, 'valid')(Input)
	x = MaxPool2D(2, 2, 'valid')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)

	x = Conv2D(16, 5, 1, 'valid')(x)
	x = MaxPool2D(2, 2, 'valid')(x)
	x = BatchNormalization()(x)
	x = Activation(tf.nn.relu)(x)

	x = Conv2D(120, 5, 1, 'valid')(x)

	x = Flatten()(x)

	x = Dense(84)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(0.2)(x)
	# x = Dense(1)(x)
	return x

