import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAveragePooling2D, \
	Flatten, Dense, Activation, BatchNormalization, Dropout, concatenate, Add


def wide_deep(Input_1, Input_2, kernel_nums: tuple = (6, 16, 120), kernel_sizes: tuple = (5, 5, 5),
              fc_sizes: tuple = (2000, 200, 1), dropout_rate: float = 0.2):
	"""
	GLCNN architecture.
	:param Input_1: output of CNN part
	:param Input_2: descriptor
	:param kernel_nums: numbers of kernels of each CNN layer
	:param kernel_sizes: sizes of kernels of  each CNN layer
	:param fc_sizes: size of each FC layer. for regression task, the last one is 1.
	:param dropout_rate: dropout
	:return: predicted values
	"""
	assert len(kernel_nums) > 1, "kernel_nums should greater than 1"
	assert len(kernel_sizes) > 1, "kernel_sizes should greater than 1"
	assert len(kernel_nums) == len(kernel_sizes), "kernel_nums and kernel_sizes should have same length"

	x_1 = Lenet5(Input_1, kernel_nums=kernel_nums, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate)
	x_2 = concatenate([x_1, Input_2])
	for size in fc_sizes[:-1]:
		x_2 = Dense(size)(x_2)
		x_2 = Activation(tf.nn.relu)(x_2)
		x_2 = Dropout(dropout_rate)(x_2)
	return Dense(1)(x_2)


def Lenet5(Input, kernel_nums, kernel_sizes, dropout_rate):
	"""
	CNN part.
	:param Input: shape (32,32)
	:param kernel_nums: numbers of kernels of each CNN layer
	:param kernel_sizes: sizes of kernels of each CNN layer
	:param dropout_rate: dropout
	:return: flatten CNN output
	"""
	for num, size in zip(kernel_nums[:-1], kernel_sizes[:-1]):
		x = Conv2D(num, size, 1, 'valid')(Input)
		x = MaxPool2D(2, 2, 'valid')(x)
		x = BatchNormalization()(x)
		x = Activation(tf.nn.relu)(x)

	x = Conv2D(kernel_nums[-1], kernel_sizes[-1], 1, 'valid')(x)
	x = Flatten()(x)
	x = Dense(84)(x)
	x = Activation(tf.nn.relu)(x)
	x = Dropout(dropout_rate)(x)
	# x = Dense(1)(x)
	return x

