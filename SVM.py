# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:11:07 2017

@author: Administrator
"""
from sklearn import svm
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

inputfile = r'xxxx'
data = pd.read_excel(inputfile)
df_train,df_test = train_test_split(data, random_state =1, test_size = 0.25)# 
y_train = df_train['travel_time'].values
y_test = df_test['travel_time'].values
x_train = df_train.drop('travel_time',axis = 1).values
x_test = df_test.drop('travel_time',axis = 1).values

clf = svm.SVR(c=1.0,epsilon=0.2)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print('the RMSE of prediction is:',mean_squared_error(y_test,y_pred)**0.5)
print('tee MAPE of prediction is:',np.mean(abs(y_pred-y_test)/y_test))