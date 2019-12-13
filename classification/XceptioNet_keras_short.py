import numpy as np
import string
import random
import os
from os.path import join
from sklearn.model_selection import train_test_split
import pickle
import argparse
import math
from datetime import datetime
import csv

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D, BatchNormalization, LSTM, concatenate, Activation, GlobalAveragePooling2D, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import Sequence, plot_model
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ReduceLROnPlateau
from matplotlib import pyplot as plt
import sklearn

import cv2
import dlib
from PIL import Image as pil_image
from tqdm import tqdm

def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve="ROC", summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def auprc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='PR', summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_model(model):
	K.clear_session()
	if model == "ResNet152":
		base_model = ResNet152(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	if model == "InceptionV3":
		base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	if model == "InceptionResNetV2":
		base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	if model == "Xception":
		base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	if model == "VGG":
		base_model = VGG19(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	base_model.summary()
	x = base_model.output
	dense1_ = Dense(512, activation='relu')
	dense1  = dense1_(x)
	x = Dropout(0.2)(dense1)
	dense2  = Dense(256, activation='relu')(x)
	x = Dropout(0.2)(dense2)
	dense3 = Dense(128, activation='relu')(x)
	pred_output = Dense(1, activation='sigmoid')(dense3)
	model = Model(inputs=base_model.input, outputs=[pred_output])
	model.summary()
	return model

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def search_files(directory='.', extension=''):
    extension = extension.lower()
    files_list = list()
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                files_list.append(os.path.join(dirpath, name))
            #elif not extension:
                #print(os.path.join(dirpath, name))
    return files_list

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
	"""
	Expects a dlib face to generate a quadratic bounding box.
	:param face: dlib face class
	:param width: frame width
	:param height: frame height
	:param scale: bounding box size multiplier to get a bigger face region
	:param minsize: set minimum bounding box size
	:return: x, y, bounding_box_size in opencv form
	"""
	x1 = face.left()
	y1 = face.top()
	x2 = face.right()
	y2 = face.bottom()
	size_bb = int(max(x2 - x1, y2 - y1) * scale)
	if minsize:
	    if size_bb < minsize:
	        size_bb = minsize
	center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

	# Check for out of bounds, x-y top left corner
	x1 = max(int(center_x - size_bb // 2), 0)
	y1 = max(int(center_y - size_bb // 2), 0)
	# Check for too big bb size for given x, y
	size_bb = min(width - x1, size_bb)
	size_bb = min(height - y1, size_bb)

	return x1, y1, size_bb

class track_generator(Sequence):

	def __init__(self, X, Y):

		self.X = X
		self.Y = Y
		if (len(X) != len(Y)):
			print("generator picture and label number does not match up")
			exit(1)
	    #find all pictures in png_dir

	def __len__(self):
		# sprint(x_len)
		return len(self.X)

	def __getitem__(self, idx):
		image = cv2.imread(self.X[idx])

		# report details about the image
		# print(type(image))
		# print(image.format)
		# print(image.mode)
		# print(image.shape)

		# convert to numpy array
		# img_array = img_to_array(image)
		# print(img_array.dtype)
		# print(img_array.shape)

		# Image size
		height, width = image.shape[:2]

		# 2. Detect with dlib
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		face_detector = dlib.get_frontal_face_detector()
		faces = face_detector(gray, 1)

		#if there are faces
		if len(faces):
			# For now only take biggest face
			face = faces[0]

			# Face crop with dlib and bounding box scale enlargement
			x, y, size = get_boundingbox(face, width, height)
			# print("size of the crop is: " + str(size))
			cropped_face = image[y:y+size, x:x+size]
			resized = cv2.resize(cropped_face, (299, 299), interpolation = cv2.INTER_AREA)
			resized = np.expand_dims(resized, axis=0)
			resized = preprocess_input(resized)
			# print(resized.shape)

			# return cropped faces
			return resized, self.Y[idx]


if __name__ == '__main__':

	#parsing command line arguments
	parser = argparse.ArgumentParser(description='enter the argument for training the CNN')
	parser.add_argument('-m', '--model', type=str, help='VGG, Xception, FlyNetTrain, FLyNetEval')
	parser.add_argument('-p', '--pos_dir', type=str, help='manipulated picture directory')
	parser.add_argument('-n', '--neg_dir', type=str, help='original picture directory')
	args = parser.parse_args()

	output_name = 'classifier.short.'+args.model
	os.system("mkdir -p ./figures/"+output_name)
	figure_output_name = './figures/' + output_name
	print(figure_output_name)

	if (args.model == "FlyNetTrain"):
		manipulated_dir = sorted(os.listdir(args.pos_dir))
		man_orig_dir = sorted([m.split("_")[0] for m in manipulated_dir])
		original_dir = sorted(os.listdir(args.neg_dir))

		if man_orig_dir != original_dir:
			print("original video list does not match manipulated video list")
			exit(1)

		#reserve data for meta data training
		metaTraining_manipulated_dir = []
		metaTraining_man_orig_dir = []
		metaTraining_original_dir = []
		for i in range(100):
			metaTraining_manipulated_dir.append(manipulated_dir.pop())
			metaTraining_man_orig_dir.append(man_orig_dir.pop())
			metaTraining_original_dir.append(original_dir.pop())

		#reserve data for meta data training
		metaTesting_manipulated_dir = []
		metaTesting_man_orig_dir = []
		metaTesting_original_dir = []
		for i in range(50):
			metaTesting_manipulated_dir.append(manipulated_dir.pop())
			metaTesting_man_orig_dir.append(man_orig_dir.pop())
			metaTesting_original_dir.append(original_dir.pop())

		metadata = open("metadata.txt", "w")
		joblist = open("joblist.txt", "w")
		for o_idx in range(len(original_dir)):
			o = original_dir[o_idx]
			m_idx = man_orig_dir.index(o)
			m = manipulated_dir[m_idx]
			writer = csv.writer(metadata, delimiter='\t')
			writer.writerows([args.pos_dir + "/" + m, #the positive values
				args.neg_dir + "/" + o,  #the negative values
				",".join(metaTesting_original_dir), #the positive values to predict on
				",".join(metaTesting_manipulated_dir), #the negative values to predict on
				o_idx]) #the index of the model based on the video pair

			joblist.write("cd /gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification; " + 
				"module load miniconda; source activate faceforensics; python FlyNet.py" +
				" -p " + args.pos_dir + #specify the directory of the positive images
				" -n " + args.neg_dir + # ... 			     of the negative images
				" -m " + m + 		    #the positive video (image set) used for training
				" -o " + o +		    #the negative ...
				" -s " + "/gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification/models" +      #storage of trained models
				" -r " + "/gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification/predictions" + #storage result of predictions
				" -a " + ",".join(metaTesting_manipulated_dir) + #prediction needed of positives
				" -b " + ",".join(metaTesting_original_dir)) 	 #prediction needed of negatives

	if (args.model == "FlyNetEval"):
		#check if the trained models are there
		dirContents = os.listdir("/gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification/models")
		if len(dirContents) == 0:
			print("there are no models, run FlyNetTrain")

		#construct meta net
		input_size = Input(shape=(1000, 1))
		conv1_ = Conv1D(128, 3, padding='same',activation='relu')(input_size)
		conv2_ = Conv1D(64, 3, padding='same',activation='relu')(conv1_)
		conv3_ = Conv1D(64, 3, padding='same',activation='relu')(conv2_)
		conv4_ = Conv1D(128, 3, padding='same',activation='relu')(conv3_)
		pool1  = MaxPooling1D(pool_size=2)(conv4_)
		conv5_ = Conv1D(64, 3, padding='same',activation='relu')(pool1)
		conv6_ = Conv1D(64, 3, padding='same',activation='relu')(conv5_)
		conv7_ = Conv1D(128, 3, padding='same',activation='relu')(conv6_)
		pool2  = MaxPooling1D(pool_size=2)(conv7_)

		x = Flatten()(pool2)
		dense1_ = Dense(126, activation='relu')
		dense1  = dense1_(x)
		x = Dropout(0.5)(dense1)
		dense2  = Dense(64, activation='relu')(x)
		x = Dropout(0.5)(dense2)
		dense3 = Dense(32, activation='relu')(x)
		pred_output = Dense(1, activation='sigmoid')(dense3)
		model = Model(input=[input_size], output=[pred_output])
		model.summary()

		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
		model.compile(loss='binary_crossentropy', optimizer=adam, 
			metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

		#use predictions from all small neural net as input to train
		#load prediction features from mini-net

		#train the model
		history = model.fit(x_train, y_train,
			batch_size=1,
			epochs=40,
			validation_split=0.0)

		#predict the results on the out of sample images
		y_pred = model.predict(x_test)
		y_pred.input_shape
		accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))

		figure_dir = "/gpfs/ysm/scratch60/gerstein/zc264/FaceForensics/classification/results/"
		os.system("mkdir -p "+figure_dir)
		figure_output_name = figure_dir + "final"

		# plot accuracy
		plt.figure()
		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig(figure_output_name+'.accuracy.png')

		# plot loss over time in cell A
		plt.figure()
		plt.plot(history.history['loss'])
		plt.title('model binary entropy loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig(figure_output_name+'.loss.png')

		# auroc over time in cell A
		plt.figure()
		plt.plot(history.history['auroc'])
		plt.title('model auROC')
		plt.ylabel('auroc')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig(figure_output_name+'.auROC.png')

		# auprc over time in cell A
		plt.figure()
		plt.plot(history.history['auprc'])
		plt.title('model auPRC')
		plt.ylabel('auprc')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig(figure_output_name+'.auPRC.png')

		# ROC in test set (cell B)
		plt.figure()
		fpr_keras, tpr_keras, thresholds_keras = sklearn.metrics.roc_curve(y_test, y_pred)
		auroc_s = sklearn.metrics.auc(fpr_keras, tpr_keras)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auroc_s))
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title('ROC curve')
		plt.legend(loc='best')
		plt.savefig(figure_output_name+'.ROC.png')

		# PRC in test set (cell B)
		plt.figure()
		precision_keras, recall_keras, thresholds_keras = sklearn.metrics.precision_recall_curve(y_test, y_pred)
		auprc_s = sklearn.metrics.auc(recall_keras, precision_keras)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(recall_keras, precision_keras, label='Keras (area = {:.3f})'.format(auprc_s))
		plt.xlabel('Precision')
		plt.ylabel('Recall')
		plt.title('PR curve')
		plt.legend(loc='best')
		plt.savefig(figure_output_name+'.PR.png')

		print(accuracy_s, auroc_s, auprc_s)

	if (args.model == "VGG") or (args.model == "Xception") or (args.model == "ResNet152") or (args.model == "InceptionV3") or (args.model == "InceptionResNetV2"):

		if args.model == "ResNet152":
			from keras.applications.resnet import preprocess_input 
		if args.model == "InceptionV3":
			from keras.applications.inception_v3 import preprocess_input 
		if args.model == "InceptionResNetV2":
			from keras.applications.inception_resnet_v2 import preprocess_input 
		if args.model == "Xception":
			from keras.applications.xception import preprocess_input
		if args.model == "VGG":
			from keras.applications.vgg19 import preprocess_input

		print(args.model)
		pos_files_list = search_files(directory=args.pos_dir, extension='.png')
		neg_files_list = search_files(directory=args.neg_dir, extension='.png')
		files_list = np.array(pos_files_list + neg_files_list)
		label_list = np.array([1] * len(pos_files_list) + [0] * len(neg_files_list))
		label_list = label_list.reshape((len(label_list), 1))
		print(files_list[0], len(files_list))
		print(label_list[0], len(label_list))

		random.seed(0)
		idx = np.array(random.sample(range(len(label_list)), 12500))
		print(idx[0:10])
		files_list = files_list[idx]
		print(files_list.shape)
		label_list = label_list[idx]
		print(label_list.shape)

		X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(files_list, label_list, test_size=0.1, random_state=42)

		train_generator = track_generator(X_train, y_train)
		test_generator = track_generator(X_test, y_test)
		pred_generator = track_generator(X_test, y_test)
		model = load_model(args.model)

		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
		model.compile(loss='binary_crossentropy', optimizer=adam, 
		    metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

		#fit the model
		history = model.fit_generator(train_generator, epochs=10, validation_data=test_generator, 
		    shuffle=False, max_queue_size=50, use_multiprocessing=True, workers=16)
		#save the model and the weights in case if the model doesn't work
		os.system("mkdir -p ./model/"+output_name)
		model.save_weights('./model/'+output_name+'/' + output_name + '.weights.h5')
		model.save('./model/'+output_name+'/' + output_name + '.h5')

		# predict the results
		# pred_step_size = 50
		y_pred = model.predict_generator(pred_generator, #steps=pred_step_size, 
			max_queue_size=50, use_multiprocessing=True, workers=16).ravel()

		#measuring the accuracy enhancer prediction in cell B
		accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))

		# plot accuracy over time in cell A
		plt.figure()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(figure_output_name+'.accuracy.png')

		# plot loss over time in cell A
		plt.figure()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model binary entropy loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(figure_output_name+'.loss.png')

		# auroc over time in cell A
		plt.figure()
		plt.plot(history.history['auroc'])
		plt.plot(history.history['val_auroc'])
		plt.title('model auROC')
		plt.ylabel('auroc')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(figure_output_name+'.auROC.png')

		# auprc over time in cell A
		plt.figure()
		plt.plot(history.history['auprc'])
		plt.plot(history.history['val_auprc'])
		plt.title('model auPRC')
		plt.ylabel('auprc')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(figure_output_name+'.auPRC.png')

		# ROC in test set (cell B)
		plt.figure()
		fpr_keras, tpr_keras, thresholds_keras = sklearn.metrics.roc_curve(y_test, y_pred)
		auroc_s = sklearn.metrics.auc(fpr_keras, tpr_keras)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auroc_s))
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title('ROC curve')
		plt.legend(loc='best')
		plt.savefig(figure_output_name+'.ROC.png')

		# PRC in test set (cell B)
		plt.figure()
		precision_keras, recall_keras, thresholds_keras = sklearn.metrics.precision_recall_curve(y_test, y_pred)
		auprc_s = sklearn.metrics.auc(recall_keras, precision_keras)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(recall_keras, precision_keras, label='Keras (area = {:.3f})'.format(auprc_s))
		plt.xlabel('Precision')
		plt.ylabel('Recall')
		plt.title('PR curve')
		plt.legend(loc='best')
		plt.savefig(figure_output_name+'.PR.png')

		print(accuracy_s, auroc_s, auprc_s)



		# #kfold division of the data
		# kf = sklearn.model_selection.KFold(n_splits=2, random_state=None, shuffle=False)

		# #gather cross validated data
		# history_list = []
		# y_pred_list = []
		# y_test_list = []
		# accuracy_list = []

		# for train_index, test_index in kf.split(label_list):
		# 	print(train_index[0],len(train_index))
		# 	print(test_index[0],len(test_index))

		# 	X_train = files_list[train_index]
		# 	X_test = files_list[test_index]
		# 	y_train = label_list[train_index]
		# 	y_test = label_list[test_index]

		# 	train_generator = track_generator(X_train, y_train)
		# 	test_generator = track_generator(X_test, y_test)
		# 	pred_generator = track_generator(X_test, y_test)
		# 	model = load_model(args.model)

		# 	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
		# 	model.compile(loss='binary_crossentropy', optimizer=adam, 
		# 	    metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

		# 	#fit the model
		# 	history_list.append(model.fit_generator(train_generator, epochs=5, validation_data=test_generator, 
		# 	    shuffle=True, max_queue_size=50, use_multiprocessing=True, workers=8))
		# 	#save the model and the weights in case if the model doesn't work
		# 	os.system("mkdir -p ./model/"+output_name)
		# 	model.save_weights('./model/'+output_name+'/' + output_name + '.weights.h5')
		# 	model.save('./model/'+output_name+'/' + output_name + '.h5')

		# 	# predict the results
		# 	# pred_step_size = 50
		# 	y_pred = model.predict_generator(pred_generator, #steps=pred_step_size, 
		# 		max_queue_size=50, use_multiprocessing=True, workers=8).ravel()
		# 	y_pred_list.append(y_pred)
		# 	print(y_pred.shape)

		# 	#y_test_sample = y_test[0:pred_step_size].ravel()
		# 	y_test_sample = y_test.ravel()
		# 	y_test_list.append(y_test_sample)
		# 	print(y_test_sample.shape)

		# 	accuracy_s = sklearn.metrics.accuracy_score(y_test_sample, np.rint(y_pred))
		# 	print(accuracy_s)
		# 	accuracy_list.append(accuracy_s)

		# 	del model

		# print(len(y_test_list), len(y_pred_list))
		# print(type(y_test_list), type(y_pred_list))

		# y_test_out = []
		# y_pred_out = []
		# for j in range(len(y_test_list)):
		# 	print(len(y_test_list[j]), len(y_pred_list[j]), type(y_test_list[j]))
		# 	y_test_out.extend(y_test_list[j])
		# 	y_pred_out.extend(y_pred_list[j])

		# # plot accuracy over time
		# plt.figure()
		# history_acc = np.array([np.array(h.history['acc']) for h in history_list])
		# history_val_acc = np.array([np.array(h.history['val_acc']) for h in history_list])
		# mean_history_acc = np.mean(history_acc, axis=0)

		# plt.plot(mean_history_acc, label='Keras (5cv_acc = {:.3f})'.format(np.mean(np.array(accuracy_list))))
		# plt.title('training and final validation accuracy')
		# plt.ylabel('accuracy')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'val'], loc='upper left')
		# plt.savefig(figure_output_name+'.accuracy.png')

		# # plot loss over time
		# plt.figure()
		# history_loss = np.array([np.array(h.history['loss']) for h in history_list])
		# mean_history_loss = np.mean(history_loss, axis=0)

		# plt.plot(mean_history_loss)
		# plt.title('training loss')
		# plt.ylabel('loss')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'val'], loc='upper left')
		# plt.savefig(figure_output_name+'.loss.png')

		# # auroc over time
		# plt.figure()
		# history_auroc = np.array([np.array(h.history['auroc']) for h in history_list])
		# mean_history_auroc = np.mean(history_auroc, axis=0)

		# plt.plot(mean_history_auroc)
		# plt.title('training auROC')
		# plt.ylabel('auroc')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'val'], loc='upper left')
		# plt.savefig(figure_output_name+'.auROC.png')

		# # auprc over time
		# plt.figure()
		# history_auprc = np.array([np.array(h.history['auprc']) for h in history_list])
		# mean_history_auprc = np.mean(history_auprc, axis=0)

		# plt.plot(mean_history_auprc)
		# plt.title('training auPRC')
		# plt.ylabel('auprc')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'val'], loc='upper left')
		# plt.savefig(figure_output_name+'.auPRC.png')


		# # ROC in test set
		# plt.figure(figsize=(5, 5))
		# base_fpr = np.linspace(0, 1, 101)
		# tpr_list = []
		# auroc_list = []
		# for i in range(len(y_test_list)):
		# 	fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_list[i], y_pred_list[i])
		# 	auroc_list.append(sklearn.metrics.roc_auc_score(y_test_list[i], y_pred_list[i]))
		# 	plt.plot(fpr, tpr, 'b', alpha=0.15)
		# 	tpr = np.interp(base_fpr, fpr, tpr)
		# 	tpr[0] = 0.0
		# 	tpr_list.append(tpr)


		# print(len(tpr_list), len(tpr_list[0]), len(tpr_list[1]))
		# tpr_list = np.array(tpr_list)
		# mean_tpr = np.mean(np.array(tpr_list), axis=0)
		# tpr_std = tpr_list.std(axis=0)

		# tprs_upper = np.minimum(mean_tpr + 2 * tpr_std, 1)
		# tprs_lower = mean_tpr - 2 * tpr_std

		# plt.plot([0, 1], [0, 1], 'k--')
		# plt.plot(base_fpr, mean_tpr, 'b', label='Keras (area = {:.3f})'.format(np.mean(np.array(auroc_list))))
		# plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
		# plt.xlabel('False positive rate')
		# plt.ylabel('True positive rate')
		# plt.title('ROC curve')
		# plt.legend(loc='best')
		# plt.axes().set_aspect('equal', 'datalim')
		# plt.savefig(figure_output_name+'.ROC.png')

		# # PRC in test set
		# plt.figure(figsize=(5, 5))
		# base_recall = np.linspace(0, 1, 101)
		# precision_list = []
		# auprc_list = []
		# for i in range(len(y_test_list)):
		# 	recall, precision, thresholds = sklearn.metrics.precision_recall_curve(y_test_list[i], y_pred_list[i])
		# 	auprc_list.append(sklearn.metrics.average_precision_score(y_test_list[i], y_pred_list[i]))
		# 	plt.plot(recall, precision, 'b', alpha=0.15)
		# 	precision = np.interp(base_recall, recall, precision)
		# 	precision[0] = 1.0
		# 	precision_list.append(precision)
		    
		# print(len(precision_list), len(precision_list[0]), len(precision_list[1]))
		# precision_list = np.array(precision_list)
		# mean_precision = np.mean(np.array(precision_list), axis=0)
		# precision_std = precision_list.std(axis=0)

		# precisions_upper = np.minimum(mean_precision + 2 * precision_std, 1)
		# precisions_lower = mean_precision - 2 * precision_std

		# plt.plot([0, 1], [1, 0], 'k--')
		# plt.plot(base_recall, mean_precision, 'b', label='Keras (area = {:.3f})'.format(np.mean(np.array(auprc_list))))
		# plt.fill_between(base_recall, precisions_lower, precisions_upper, color='grey', alpha=0.3)
		# plt.xlabel('recall')
		# plt.ylabel('precision')
		# plt.title('PRC curve')
		# plt.legend(loc='best')
		# plt.axes().set_aspect('equal', 'datalim')
		# plt.savefig(figure_output_name+'.PRC.png')

		# print(np.mean(np.array(accuracy_list)), np.mean(np.array(auroc_list)), np.mean(np.array(auprc_list)))





