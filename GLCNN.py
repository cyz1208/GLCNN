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
import argparse

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
	load DA graphs, grids and outputs
	"""
	clean_name = data_clean('./structure.log')

	# DA grid
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

	# DA graph and descriptor
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

	# DA outputs
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


def data_process(repeat):
	"""
	split train, validation and test set.
	"""
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
	rand_index = np.random.choice(list(range(repeat)), size=repeat, replace=False)
	aug_train_index, aug_val_index, aug_test_index = [], [], []
	for _i in train_index:
		for _j in rand_index:
			aug_train_index.append(_i * repeat + _j)
	for _i in val_index:
		for _j in rand_index:
			aug_val_index.append(_i * repeat + _j)
	for _i in test_index:
		for _j in rand_index:
			aug_test_index.append(_i * repeat + _j)

	# Generate training set and test set after DA
	X_train_1, X_train_2, aug_y_train = X_1[aug_train_index], X_2[aug_train_index], aug_y[aug_train_index]
	X_val_1, X_val_2, aug_y_val = X_1[aug_val_index], X_2[aug_val_index], aug_y[aug_val_index]
	X_test_1, X_test_2, aug_y_test = X_1[aug_test_index], X_2[aug_test_index], aug_y[aug_test_index]
	print(f"train: {X_train_1.shape}, {X_train_2.shape}, {aug_y_train.shape}.")
	print(f"val: {X_val_1.shape}, {X_val_2.shape}, {aug_y_val.shape}.")
	print(f"test: {X_test_1.shape}, {X_test_2.shape}, {aug_y_test.shape}.")

	# Extract the original output from the test set output after DA
	y_origin = aug_y_test.reshape(-1, repeat)[:, 0]
	return X_train_1, X_train_2, aug_y_train, X_val_1, X_val_2, aug_y_val,\
		X_test_1, X_test_2, aug_y_test, y_origin


def build_model():
	"""
	model construction
	"""
	Input_1 = tf.keras.layers.Input(shape=(32, 32, 6))
	Input_2 = tf.keras.layers.Input(shape=(15,))
	Output = CNN.wide_deep(Input_1, Input_2)
	model = tf.keras.Model(inputs=[Input_1, Input_2], outputs=[Output])
	# model.summary()
	opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=opt, loss='mse', metrics=['mae'])
	return model


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


def train_model(model, X_train, y_train, X_val, y_val, epoch, batch, model_ckpt, log_dir):
	"""
	train model and save as ckpt file
	"""
	warmup_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

	start = time.perf_counter()
	model.fit(X_train, y_train,
			batch_size=batch,
			epochs=epoch, verbose=2, shuffle=True,
			# validation_split=0.2,
			validation_data=(X_val, y_val),
			callbacks=[
			tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5),
			# tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=2),
			tf.keras.callbacks.ModelCheckpoint(model_ckpt, save_best_only=True, verbose=2, monitor="val_mae"),
			# tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5,
				# min_lr=0.0001, verbose=2, monitor="val_loss"),
			warmup_lr,
	])
	end = time.perf_counter()
	print(f"model train time consumed: {np.round(end - start, decimals=3)} sec.")


def test_model(X_test, y_test, y_origin, model_ckpt, repeat):
	"""
	load model from ckpt file, test model and save the results
	"""
	model = tf.keras.models.load_model(model_ckpt)
	y_pred = model.predict(X_test)

	# MAE and RMSE
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	print(f"MAE: {mae}, RMSE: {np.sqrt(mse)}.")

	# average of std of DA samples derived from the same original sample
	y_sets = y_pred.reshape(-1, repeat)   # each set contains predicted value of DA samples from same original one
	y_sets_std = np.std(y_sets, axis=1)
	print(f"DA samples std: {np.average(y_sets_std)}.")

	# MAE* and RMSE*
	y_pred = np.average(y_sets, axis=-1)
	mae = mean_absolute_error(y_origin, y_pred)
	mse = mean_squared_error(y_origin, y_pred)
	print(f"MAE*: {mae}, RMSE*: {np.sqrt(mse)}.")

	# save predicted and true values
	df = pd.DataFrame(np.array([y_pred, y_origin]).T, columns=["pred", "test"])
	df.to_csv("./out_OH.csv", index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--batch", type=int, default=256, help="batch size")
	parser.add_argument("-r", "--repeat", type=int, default=20, help="DA iterations with maximum of 20")
	parser.add_argument("-e", "--epoch", type=int, default=200, help="epoch of model training")
	args = parser.parse_args()

	ROOT_DIR = os.getcwd()
	LOG_DIR = os.path.join(ROOT_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	MODEL_CKPT = os.path.join(ROOT_DIR, "model_opt")
	BATCH_SIZE = args.batch
	REPEAT = args.repeat
	EPOCH = args.epoch

	X_train_1, X_train_2, aug_y_train, X_val_1, X_val_2, aug_y_val,\
		X_test_1, X_test_2, aug_y_test, y_origin = data_process(REPEAT)

	model = build_model()
	train_model(model, [X_train_1, X_train_2], aug_y_train,
	            [X_val_1, X_val_2], aug_y_val,
	            epoch=EPOCH, batch=BATCH_SIZE, model_ckpt=MODEL_CKPT, log_dir=LOG_DIR)

	test_model([X_test_1, X_test_2], aug_y_test, y_origin,
	           model_ckpt=MODEL_CKPT, repeat=REPEAT)


