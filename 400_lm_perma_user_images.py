#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:40:44 2019

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


ml_y_var = 'y_perma'

# Get the list of columns that can be used for ML.  
ml_X_cols = []

ml_X_cols = ['x_user_follows','x_image_count','x_images_mean_comment_count','x_images_mean_like_count']

# Use the next block if you want to include a bunch of columns
#for col in df.columns:
#    if col[0:2] in ['x_']:
#        ml_X_cols.append(col)

# Toast the last few nulls 
df=df.dropna()

# Split the data 70/30 into train test 
df_train, df_test = train_test_split(df, test_size=0.3, random_state=73)

# Run linear regression model!
lrm = LinearRegression().fit(X=df_train[ml_X_cols], y=df_train[ml_y_var])

# Predict on the test set!
# Add the prediction variable to the test set
df_test[ml_y_var + '_pred'] = lrm.predict(df_test[ml_X_cols])

lrm_score = lrm.score(X=df_train[ml_X_cols], y=df_train[ml_y_var])

lrm_rms = sqrt(mean_squared_error(df_test[ml_y_var], df_test[ml_y_var + '_pred']))

lrm_r2s = r2_score(df_test[ml_y_var], df_test[ml_y_var + '_pred'])

print(lrm_score)
print(lrm_rms)
print(lrm_r2s)




from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

X = df_train[ml_X_cols]
y = df_train[ml_y_var]


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

