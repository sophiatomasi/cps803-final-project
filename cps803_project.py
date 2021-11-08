#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:35:15 2021

@author: elwadgedleh
@author: sophiatomasi
"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

heart_data = pd.read_csv("/Users/elwadgedleh/Downloads/heart.csv")
heart_data = heart_data.drop('restecg', 1)
heart_data = heart_data.drop('slp',1)
heart_data = heart_data.drop('caa',1)
heart_data = heart_data.drop('thall',1)
print(heart_data)

train_data, test_data = train_test_split(heart_data, test_size=0.2, random_state=42, shuffle=True)

x_train = train_data.iloc[:,0:9]
y_train = train_data.iloc[:,9]

print(x_train)
print(y_train)

x_test = test_data.iloc[:,0:9]
y_test = test_data.iloc[:,9]

print(x_test)
print(y_test)
