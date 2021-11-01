#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:35:15 2021

@author: elwadgedleh
@author: sophiatomasi
"""
import pandas as pd 
import matplotlib.pyplot as plt


heart_data = pd.read_csv("/Users/elwadgedleh/Downloads/heart.csv")
heart_data = heart_data.drop('restecg', 1)
#heart_data = heart_data[heart_data.sex != 1]
x_train = heart_data.iloc[:,0:9]
print(x_train)
