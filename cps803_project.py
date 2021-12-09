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
from sklearn import metrics
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

heart_data = pd.read_csv("heart.csv")
heart_data = heart_data.drop('restecg', 1)
heart_data = heart_data.drop('slp',1)
heart_data = heart_data.drop('caa',1)
heart_data = heart_data.drop('thall',1)
print(heart_data)

#Recoding Continuos Variables => Age, Resting blood pressure, Cholesteral, Max heart rate
category_age = pd.cut(heart_data.age,bins=[18,29,41,53,65,77,100],labels=[0,1,2,3,4,5])
category_age=category_age.astype('float64')
heart_data.insert(0,'Age-Group',category_age)
heart_data= heart_data.drop('age',1)

category_trtbps = pd.cut(heart_data.trtbps,bins=[90,120,140,160,200],labels=[0,1,2,3])
category_trtbps=category_trtbps.astype('int64')
heart_data.insert(3,'Resting-BP-Group',category_trtbps)
heart_data= heart_data.drop('trtbps',1)

category_chol = pd.cut(heart_data.chol,bins=[125,199,239,600],labels=[0,1,2])
category_chol=category_chol.astype('int64')
heart_data.insert(4,'Chol',category_chol)
heart_data= heart_data.drop('chol',1)

category_thalachh = pd.cut(heart_data.thalachh,bins=[60,110,125,135,145,155,165,175,185,220],
                            labels=[0,1,2,3,4,5,6,7,8])
category_thalachh=category_thalachh.astype('int64')
heart_data.insert(6,'Max-HR',category_thalachh)
heart_data= heart_data.drop('thalachh',1)

#Removing data points: 303 rows, drop 150 
#heart_data = heart_data.drop(heart_data.index[range(150)],axis=0)

#Dropping feature with highest dependancy 
#heart_data = heart_data.drop('Age-Group',1)

train_data, test_data = train_test_split(heart_data, test_size=0.3, random_state=42, shuffle=True)

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

# ---------------------------Logistic Regression------------------------------------------
lr_model = LogisticRegression(random_state=0,max_iter=1000).fit(x_train, y_train)
#lr_model = LogisticRegression(C=.3,random_state=0,max_iter=1000).fit(x_train, y_train)
#Store predictions
predicted_labels_lr = lr_model.predict(x_test)
#Overall accuracy of test set
score_lr=lr_model.score(x_test,y_test) 
print("Score of LR: ", score_lr)


predicted_labels_lr_train = lr_model.predict(x_train)
score_lr_train = lr_model.score(x_train,y_train)
print("Score of LR Train:", score_lr_train)

#Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, predicted_labels_lr)
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="icefire" )
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('logistic_regression_confusion_matrix')
plt.show()
plt.close()

#Check for dependant features using VIF 
vif_data = pd.DataFrame()  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.values.shape[1])]
vif_data["features"] = x_train.columns
print(vif_data)
# -------------------------------------------------------------------------------------------------
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
