import functools
import os
import numpy as np
import pickle
import pandas as pd
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error, mean_squared_error
from graph import Utils
import datetime
import CNN
import matplotlib.pyplot as plt

# tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only allocate 2GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(
			gpus[0],
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)],
		)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)


def timer(func):
	@functools.wraps(func)
	def inner(*args, **kargs):
		t0 = time.perf_counter()
		res = func(*args, **kargs)
		t1 = time.perf_counter()
		print(f"func {func.__name__} consumed time: {np.round(t1 - t0, decimals=3)} sec.")
		return res
	return inner


def read_csv(filename) -> pd.DataFrame:
	print('read csv ...')
	df = pd.read_csv(filename)
	return df


def data_clean(*filenames):
	"""
	read name of unstable structures
	"""
	print('data cleaning ...')
	name = []
	for filename in filenames:
		with open(filename, 'r') as f:
			data = f.readlines()
		for l in data:
			name.append(l.split('	')[:-1])
	return name


def data_augmentation_y(y, times=20):
	"""
	input shape: [batch, ...], return shape: [batch * times, ...]
	"""
	y_tmp = []
	for i in y:
		for _ in range(times):
			y_tmp.append(i)
	return np.array(y_tmp)


def average_pred_y(pred_y):
	"""
	average MAE of augmented data
	input shape: [batch * 20], return shape: [batch]
	"""
	return np.average(pred_y.reshape((-1, 20)), axis=-1)


@timer
def load_data():
	"""
	load graphs, grids and outputs
	"""
	clean_name = data_clean('./structure.log')

	# grid
	X_1 = []
	with open('./data/pixels.pkl', 'rb') as f:
		pixels = pickle.load(f)
	for name, pixel in pixels:
		# data clean
		if name.split() in clean_name:
			continue
		if '' in name:
			X_1.append(pixel)
	print(f"total X_1 data: {np.shape(X_1)}")

	# graph and descriptor
	X_2 = []
	G = Utils.load_graphs('./data/graphs.pkl')
	for g in G:
		if g.name.split() in clean_name:
			continue
		if '' in g.name:
			X_2.append(Utils.get_shells(g))
	X_2 = np.array(X_2)
	X_2 = scale(X_2)
	X_2 = data_augmentation_y(X_2)
	print(f"total X_2 data: {np.shape(X_2)}")

	# outputs
	y = []
	db = read_csv("./E_OH_all.csv")
	for _, datum in db.iterrows():
		# data clean
		if [datum['mesh'], datum['add_N'], datum['sub'], datum['metal']] in clean_name:
			continue
		if '' in datum['mesh']:
			y.append(datum['E_ads_H'])
	y = data_augmentation_y(y)
	print(f"total y data: {np.shape(y)}")
	return np.array(X_1), np.array(X_2), np.array(y)


if __name__ == '__main__':
	root_dir = os.getcwd()
	rundir = os.path.join(root_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	model_ckpt = os.path.join(root_dir, "model_opt")
	BATCH_SIZE = 256
	REPEAT = 20   # DA times

	# load data
	with tf.device('/CPU:0'):
		X_1, X_2, aug_y = load_data()
	# X_1 = X_1[:, :, :, [0, 1, 2, 4, 5]]  # select desired channels

	origin_len = int(len(aug_y) / 20)
	index = np.array(list(range(origin_len)))

	# Randomly generate the index of train, val and test set
	train_index, test_index = train_test_split(index, test_size=0.2, random_state=42, shuffle=True)
	val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=42, shuffle=True)

	# Generate the index of train, val and test set after DA
	rand_index = np.random.choice(list(range(20)), size=REPEAT, replace=False)
	aug_train_index, aug_val_index, aug_test_index = [], [], []
	for _i in train_index:
		for _j in rand_index:
			aug_train_index.append(_i * 20 + _j)
	for _i in val_index:
		for _j in rand_index:
			aug_val_index.append(_i * 20 + _j)
	for _i in test_index:
		for _j in rand_index:
			aug_test_index.append(_i * 20 + _j)

	# Generate training set and test set after DA
	X_train_1, X_train_2, aug_y_train = X_1[aug_train_index], X_2[aug_train_index], aug_y[aug_train_index]
	X_val_1, X_val_2, aug_y_val = X_1[aug_val_index], X_2[aug_val_index], aug_y[aug_val_index]
	X_test_1, X_test_2, aug_y_test = X_1[aug_test_index], X_2[aug_test_index], aug_y[aug_test_index]
	print(f"train: {X_train_1.shape}, {X_train_2.shape}, {aug_y_train.shape}.")
	print(f"val: {X_val_1.shape}, {X_val_2.shape}, {aug_y_val.shape}.")
	print(f"test: {X_test_1.shape}, {X_test_2.shape}, {aug_y_test.shape}.")

	# Grid visualization
	# plt.imshow(X_train_1[0, :, :, 0], cmap='viridis')
	# plt.show()

	# Extract the original output from the test set output after DA
	y_test = aug_y_test.reshape(-1, REPEAT)[:, 0]

	# model construction
	Input_1 = tf.keras.layers.Input(shape=(32, 32, 6))
	Input_2 = tf.keras.layers.Input(shape=(15,))
	# global and local
	Output = CNN.wide_deep(Input_1, Input_2)
	model = tf.keras.Model(inputs=[Input_1, Input_2], outputs=[Output])
	model.summary()
	opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=opt, loss='mse', metrics=['mae'])

	# model training
	import time

	def scheduler(epoch, lr):
		"""
		Customize the learning rate including warmup
		"""
		print(f"epoch: {epoch}, learning rate: {np.round(lr, decimals=6)}.")
		warmup_steps = 20
		if epoch < warmup_steps:
			# return 0.0001  # constant warmup
			return 0.0001 + 0.0009 * (epoch + 1) / warmup_steps  # Linear increase warmup
		elif epoch == warmup_steps:
			return 0.001
		elif epoch > warmup_steps:
			# return max(lr * tf.math.exp(-0.10), 0.0001) if epoch % 5 == 0 else lr   # Stepped exponential reduction
			return max(lr * tf.math.exp(-0.015), 0.0001)  # Gradual exponential reduction
		else:
			print("lr scheduler error.")
			exit()

	warmup_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

	start = time.perf_counter()
	model.fit([X_train_1, X_train_2], aug_y_train,
				batch_size=BATCH_SIZE,
				epochs=1, verbose=2, shuffle=True,
				# validation_split=0.2,
				validation_data=([X_val_1, X_val_2], aug_y_val),
				callbacks=[
				tf.keras.callbacks.TensorBoard(log_dir=rundir, histogram_freq=5),
				# tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=2),
				tf.keras.callbacks.ModelCheckpoint(model_ckpt, save_best_only=True, verbose=2, monitor="val_mae"),
				# tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001, verbose=2, monitor="val_loss"),
				warmup_lr,
	])
	end = time.perf_counter()
	print(f"model train time consumed: {np.round(end - start, decimals=3)} sec.")

	# model test
	model = tf.keras.models.load_model(model_ckpt)
	y_pred = model.predict([X_test_1, X_test_2])

	# MAE and RMSE
	mae = mean_absolute_error(aug_y_test, y_pred)
	mse = mean_squared_error(aug_y_test, y_pred)
	print(f"MAE: {mae}, RMSE: {np.sqrt(mse)}.")

	# std of DA samples derived from 1 original sample
	y_sets = y_pred.reshape(-1, REPEAT)
	y_sets_std = np.std(y_sets, axis=1)
	print(f"DA samples std: {np.average(y_sets_std)}.")

	# MAE* and RMSE*
	y_pred = np.average(y_pred.reshape(-1, REPEAT), axis=-1)
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	print(f"MAE*: {mae}, RMSE*: {np.sqrt(mse)}.")

	# save predicted and true values
	df = pd.DataFrame(np.array([y_pred, y_test]).T, columns=["pred", "test"])
	df.to_csv("./out_OH.csv", index=False)

