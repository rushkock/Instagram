#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:51:52 2019

@author: hduser
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  r2_score
from math import sqrt

df = pd.read_pickle("survey_user_mrg.pickle")

df = pd.get_dummies(df, columns=["X_gender",'X_employed','X_income'])
# The columns added for X_employed is ugly (have a look)
# clean them up a bit
df.columns = df.columns.str.replace(' ', '_').str.lower()
df.columns = df.columns.str.replace('.', '_').str.lower()
df.columns = df.columns.str.replace(',', '').str.lower()
df.columns = df.columns.str.replace('&', '').str.lower()
df.columns = df.columns.str.replace('$', '').str.lower()
df.columns = df.columns.str.replace('(', '').str.lower()
df.columns = df.columns.str.replace("'", '').str.lower()
df.columns = df.columns.str.replace(')', '').str.lower()
df.columns = df.columns.str.replace('__', '_').str.lower()
df.columns = df.columns.str.replace('__', '_').str.lower()


ml_y_col = 'y_bool'

# Get the list of columns that can be used for ML.  
ml_X_cols = []

ml_X_cols = ['x_user_follows','x_image_count','x_images_mean_comment_count','x_images_mean_like_count']

# Use the next block if you want to include a bunch of columns
#for col in df.columns:
#    if col[0:2] in ['x_']:
#        ml_X_cols.append(col)

# Toast the last few nulls 
df=df.dropna()

df_train, df_test = train_test_split(df, test_size=.3, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1', C = .1,random_state = 1)
logreg_model = lr.fit(X=df_train[ml_X_cols],y=df_train[ml_y_col])
# Get predictions for the test set
df_test[ml_y_col + "_pred"] =logreg_model.predict(X=df_test[ml_X_cols])

from sklearn.metrics import accuracy_score
print("Accuracy Score - {}".format(accuracy_score(df_test[ml_y_col], df_test[ml_y_col + "_pred"])))
print("Score - {}".format(lr.score(X=df_test[ml_X_cols],y=df_test[ml_y_col + '_pred'])))

