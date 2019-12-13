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

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
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
import keras_metrics as km
import sklearn

import cv2
import dlib
import torch
import torch.nn as nn
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
	if model == "Xception":
	    base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	    x = base_model.output
	    dense1_ = Dense(512, activation='relu')
	    dense1  = dense1_(x)
	    x = Dropout(0.11)(dense1)
	    dense2  = Dense(256, activation='relu')(x)
	    x = Dropout(0.47)(dense2)
	    dense3 = Dense(128, activation='relu')(x)
	    pred_output = Dense(1, activation='sigmoid')(dense3)
	    model = Model(inputs=base_model.input, outputs=[pred_output])
	if model == "VGG":
	    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="max")
	    x = base_model.output
	    dense1_ = Dense(512, activation='relu')
	    dense1  = dense1_(x)
	    x = Dropout(0.11)(dense1)
	    dense2  = Dense(256, activation='relu')(x)
	    x = Dropout(0.47)(dense2)
	    dense3 = Dense(128, activation='relu')(x)
	    pred_output = Dense(1, activation='sigmoid')(dense3)
	    model = Model(inputs=base_model.input, outputs=[pred_output])
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

class track_generator(Sequence):

    def __init__(self, x, y):

    	self.x = x
    	self.y = y
    	if (len(x) != len(y)):
    		print("generator picture and label number does not match up")
    		exit(1)
        #find all pictures in png_dir

    def __len__(self):
        # sprint(x_len)
        return len(x)

    def __getitem__(self, idx):
    	image = load_img(x[idx])

		# report details about the image
		print(type(image))
		print(image.format)
		print(image.mode)
		print(image.size)

		# convert to numpy array
		img_array = img_to_array(img)
		print(img_array.dtype)
		print(img_array.shape)

		# Image size
        height, width = img_array.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)

        #if there are faces
		if len(faces):
			# For now only take biggest face
			face = faces[0]

			# Text and bb
			x = face.left()
			y = face.top()
			w = face.right() - x
			h = face.bottom() - y
			label = 'fake' if prediction == 1 else 'real'

			# Face crop with dlib and bounding box scale enlargement
			x, y, size = get_boundingbox(face, width, height)
			cropped_face = image[y:y+size, x:x+size]

			# return cropped faces
			return cropped_face, y[idx]


if __name__ == '__main__':


    #parsing command line arguments
    parser = argparse.ArgumentParser(description='enter the argument for training the CNN')
    parser.add_argument('-p', '--pos_dir', type=str, help='specify the directory of the positive images')
    parser.add_argument('-n', '--neg_dir', type=str, help='specify the directory of the negative images')
    parser.add_argument('-m', '--pos_img', type=str, help='the positive video (image set) used for training')
    parser.add_argument('-o', '--neg_img', type=str, help='the negative video (image set) used for training')
    parser.add_argument('-s', '--mod_stor', type=str, help='directory to store trained models')
    parser.add_argument('-r', '--res_stor', type=str, help='directory to storage result of predictions')
    parser.add_argument('-a', '--pos_pred', type=str, help='prediction needed of positives')
    parser.add_argument('-b', '--neg_pred', type=str, help='prediction needed of negatives')
    args = parser.parse_args()

    #load in training images as x_train, y_train
    input_size = Input(shape=(299, 299, 3))
    conv1_ = Conv2D(128, (3, 3), padding='same',activation='relu')(input_size)
    conv2_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv1_)
    conv3_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv2_)
    conv4_ = Conv2D(128, (3, 3), padding='same',activation='relu')(conv3_)
    pool1  = MaxPooling2D(pool_size=(1, 2))(conv4_)
    conv5_ = Conv2D(64, (3, 3), padding='same',activation='relu')(pool1)
    conv6_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv5_)
    conv7_ = Conv2D(128, (3, 3), padding='same',activation='relu')(conv6_)
    pool2  = MaxPooling2D(pool_size=(1, 2))(conv7_)

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


    #train the model
    history = model.fit(x_train, y_train,
        batch_size=1,
        epochs=40,
        validation_split=0.0)


    #load in the images to predict

    #predict the results on the out of sample images
    y_pred = model.predict(x_test)
    y_pred.input_shape
    accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))

    #store output

    #store model

