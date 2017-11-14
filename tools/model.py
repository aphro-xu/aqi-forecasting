from __future__ import (absolute_import, division, print_function)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Model
from keras.layers.core import Dense, Dropout
from keras.layers import Input
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.regularizers import l2
from keras.utils import np_utils

import numpy as np
import pandas as pd

data_file = "./data_set/pt.csv"
input_size = 5
output_size = 8
time_steps = 24
cell_size = 20
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

def add_layer(inpt, dropout_rate=0.5, bk_size=100, weight_decay=1E-4):
    x = Dense(bk_size, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(inpt)
    x = Dropout(rate=dropout_rate)(x)
    return x

def resblock(x, dropout_rate=0.5,bk_size=50, weight_decay=1E-4):
    out = add_layer(x, dropout_rate, bk_size, weight_decay)
    out = add([out, x])
    out = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(out)
    out = Dropout(rate=dropout_rate)(out)
    return out

inpt = Input(shape=(input_size,))

x = add_layer(inpt, dropout_rate=0.5, bk_size=50, weight_decay=1E-4)

# resblock 1
x = resblock(x, dropout_rate=0.5, bk_size=50, weight_decay=1E-4)

# transition block
x = add_layer(x, bk_size=100)

# resblock 2
x = resblock(x, bk_size=100)

# output
out = add_layer(x, bk_size=output_size)

model = Model(inpt, out)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=50, batch_size=1000)
score = model.evaluate(test_X, test_y)
print(score)