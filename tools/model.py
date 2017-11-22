#!/usr/bin/env python3
# coding=utf-8

from __future__ import (absolute_import, division, print_function)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, LSTM, TimeDistributed
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.regularizers import l2
from keras.utils import np_utils

import numpy as np
import pandas as pd

data_file = "./data_set/pt.csv"
input_size = 5
output_size = 8
batch_size = 200

cols = ['aqi', 'SO2', 'NO2', 'P10', 'P2.5', 'O3', 'CO', 'Humi-R', 'W-Direc', 'W-Speed', 'A-Pres', 'A-Temp', 'date', 'time']
label_cols = 'aqi'
weather_cols = ['Humi-R', 'W-Direc', 'W-Speed', 'A-Pres', 'A-Temp']
pollution_cols = ['SO2', 'NO2', 'P10', 'P2.5', 'O3', 'CO', 'date', 'time']

df = pd.read_csv(data_file, sep=",", names=cols, skipinitialspace=True, skiprows=1, skip_blank_lines=1, engine="python")
# print(AQILevel(df['aqi']))

# Get labels and features
labels = df[label_cols].values
df.drop(label_cols, axis=1, inplace=True)

for _, pollution in enumerate(pollution_cols):
    df.drop(pollution, axis=1, inplace=True)

features = df.values

def AQILevel(aqi):

    level = {50: 0, 100: 1, 150: 2, 200: 3,
             300: 4, 400: 5, 500: 6, 501: 7}
    index_i = [0, 50, 100, 150, 200, 300, 400, 500, 501]
    for i in index_i:
        if i >= aqi:
            break
    return level[i]

# turn aqi into category ones
for i in range(len(df)):
    labels[i] = AQILevel(labels[i])


# Get train data and test data
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=1000)

# Data preprocessed
normalizer = preprocessing.Normalizer(norm='l2').fit(train_X)
train_X = normalizer.transform(train_X)
test_X = normalizer.transform(test_X)

train_y = np_utils.to_categorical(train_y, num_classes=output_size)
test_y = np_utils.to_categorical(test_y, num_classes=output_size)


def add_layer(inpt, bk_size=100, dropout_rate=None, weight_decay=1E-4):
    x = Dense(bk_size, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(inpt)
    if dropout_rate is not None:
        x = Dropout(rate=dropout_rate)(x)
    return x


def transition_block(x, bk_size=100, dropout_rate=0.5, weight_decay=1E-4):
    x = Dense(bk_size, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(rate=dropout_rate)(x)
    x = BatchNormalization(axis=-1, momentum=0.99, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    return x


def resblock(x, nb_layers, dropout_rate=None, bk_size=100, weight_decay=1E-4):
    feature_list = [x]
    for i in range(nb_layers):
        x = add_layer(x, dropout_rate, bk_size, weight_decay)
        feature_list.append(x)
        x = Add()(feature_list)
    x = BatchNormalization(axis=-1, momentum=0.99, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    bk_size = int(bk_size/2)
    return x, bk_size

def create_resnet(nb_classes, input_size, depth=40, nb_res_block=3, bk_size=100,
                  dropout_rate=None, weight_decay=1E-4):
    inpt = Input(shape=(input_size,))
    assert (depth-4) % 3 == 0

    # layers in each res block
    nb_layers = int((depth-4)/3)

    # initial input
    x = add_layer(inpt, dropout_rate=dropout_rate, bk_size=bk_size)
    x = BatchNormalization(axis=-1, momentum=0.99, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    # Add res block
    for block_idx in range(nb_res_block-1):
        x, bk_size = resblock(x, nb_layers, bk_size)
        x = transition_block(x, bk_size=bk_size, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last res block doesnot have a transition block
    x, bk_size = resblock(x, nb_layers, bk_size, dropout_rate, weight_decay)
    x = Activation('relu')(x)

    # softmax layer
    x = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    resnet = Model(inputs=inpt, outputs=x)

    return resnet

model = create_resnet(nb_classes=output_size, input_size=input_size, depth=40, bk_size=64, dropout_rate=0.5)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=100, batch_size=1000)
score = model.evaluate(test_X, test_y)
print('\n', score)
