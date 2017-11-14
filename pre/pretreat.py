#!/usr/bin/env python3
# coding=utf-8

from __future__ import (division, absolute_import, print_function, unicode_literals)

import pandas as pd
import numpy as np
import re
from aqi import AQI

import os.path as osp

pre_path = "data_set/"

file_path = "2015_2016pt.csv"
describe_path = "pt.csv"

read_path = osp.join(pre_path, file_path)
save_path = osp.join(pre_path, describe_path)

cols = ['no', 'time', 'SO2', 'NO2', 'P10', 'P2.5', 'O3', 'CO', 'Humi-R', 'W-Direc', 'W-Speed', 'A-Pres', 'A-Temp']
df = pd.read_csv(read_path, sep=',', names=cols, skipinitialspace=True, skiprows=1, engine='python')

# fillna
for _, col in enumerate(cols):
    df[col].fillna(method='bfill', inplace=True)

df.drop('no', axis=1, inplace=True)

# split date and time
dates = []
times = []
for i in range(len(df['time'])):
    dt = re.split(r'\s+', df['time'][i])
    dates.append(dt[0])
    # times.append(dt[1].split(":")[0])
    times.append(dt[1])

# reconstruct data
df.drop('time', axis=1, inplace=True)
df['date'] = dates
df['time'] = times

# replace abnormal data
three_quart = [0.016, 0.069, 0.099, 0.065, 0.077, 1.090, 89.958, 190.723, 1.560, 1021.571, 25.194]
max_right = np.dot(2.5, three_quart)
num_cols = ['SO2', 'NO2', 'P10', 'P2.5', 'O3', 'CO', 'Humi-R', 'W-Direc', 'W-Speed', 'A-Pres', 'A-Temp']

for idx, col in enumerate(num_cols):
    df[col] = np.clip(df[col], df[col].min(), max_right[idx])

# calculate aqi
aqis = []
for i in range(len(df)):
    aqis.append(AQI(df['P2.5'][i]*1000, df['P10'][i]*1000, df['CO'][i]*100,
                    df['SO2'][i]*1000, df['NO2'][i]*1000, df['O3'][i]*1000))

df.insert(0, 'aqi', aqis)

df.to_csv(save_path, sep=',', index=False, header=1, float_format=np.float32)