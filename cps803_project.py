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
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

heart_data = pd.read_csv("heart.csv")
heart_data = heart_data.drop('restecg', 1)
heart_data = heart_data.drop('slp',1)
heart_data = heart_data.drop('caa',1)
heart_data = heart_data.drop('thall',1)
print(heart_data)

#Removing data points: 303 rows, drop 150 
#heart_data = heart_data.drop(heart_data.index[range(150)],axis=0)

train_data, test_data = train_test_split(heart_data, test_size=0.2, random_state=42, shuffle=True)

x_train = train_data.iloc[:,0:9]
y_train = train_data.iloc[:,9]

print(x_train)
print(y_train)

x_test = test_data.iloc[:,0:9]
y_test = test_data.iloc[:,9]

print(x_test)
print(y_test)

# using Gaussian Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
score_gnb = gnb.score(x_test, y_test)
print("Score of GNB: ", score_gnb)
print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred).sum()))

# using Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
score_mnb = mnb.score(x_test, y_test)
print("Score of MNB: ", score_mnb)
print("Predicted target values for x: ", mnb.predict(x_test))

lr_model = LogisticRegression(random_state=0,max_iter=1000).fit(x_train, y_train)
#lr_model = LogisticRegression(C=.3,random_state=0,max_iter=1000).fit(x_train, y_train)
predicted_labels_lr = lr_model.predict(x_test)
score_lr=lr_model.score(x_test,y_test) 
print(score_lr)

bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(x_train,y_train)
predicted_labels_nb=bernoulli_nb_model.predict(x_test)
score_bnb=bernoulli_nb_model.score(x_test,y_test) 
print(score_bnb)

# using Decision Tree
dtc = DecisionTreeClassifier(random_state=0)
dtc_score = cross_val_score(dtc, x_train, y_train, cv=10)
dtc_accuracy = np.mean(dtc_score)
print("Cross val score: ", dtc_score)
print("DTC accuracy: ", dtc_accuracy)
