#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:20:05 2019

@author: hduser
"""

import pandas as pd

survey = pd.read_pickle("survey.pickle")
s = survey.query('insta_user_id == "263042348"')

anp = pd.read_pickle("anp.pickle")
celebrity = pd.read_pickle("celebrity.pickle")
face = pd.read_pickle("face.pickle")
image_data = pd.read_pickle("image_data.pickle")
image_metrics = pd.read_pickle("image_metrics.pickle")
object_labels = pd.read_pickle("object_labels.pickle")


# start with the survey data.  Take a nice neat subset of that

survey_ss = survey['insta_user_id','gender',']


idata = image_data.query('image_id == "1041654001515500189_703978203"')

im = image_metrics.query('image_id == "429765682836845999_263042348"')

ol = object_labels.query('image_id == "429765682836845999_263042348"')

f = face.query('image_id == "429765682836845999_263042348"')

image_data.dtypes

263042348
3988856
1429720420
53918317
1441644483
11520833
