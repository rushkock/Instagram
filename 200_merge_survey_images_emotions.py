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
# this is the survey data - 160 ppl roughly
survey_ss = survey[['insta_user_id','gender','income','PERMA']]


# these datasets have 1000's of pictures for the 160 users.  It needs to be summarized
image_data_ss = image_data[['user_id','user_name','image_filter','image_id','image_link','user_followed_by','user_follows','user_posted_photos']]
faces_ss = face[['image_id','face_id','face_smile','face_smile_confidence','face_emo','emo_confidence']]

# k the faces dataset has the most data (ie multiple rows per face per picture)
# first summarize it per picture
faces_image_sumr = faces_ss.groupby(by='image_id').agg({"face_id":pd.Series.nunique,"face_smile_confidence":pd.Series.mean})

faces_image_sumr.rename(columns={"face_id":"face_count","face_smile_confidence":"smile_confidence_mean"},inplace=True) 

faces_smile_sumr = faces_image_sumr.groupby(by="image_id").agg({"smile_confidence_mean":pd.Series.mean})

faces_emotions_sumr = pd.crosstab(faces_ss.image_id, faces_ss.face_emo,values=faces_ss.emo_confidence,aggfunc=pd.Series.mean)
faces_emotions_sumr.fillna(0,inplace=True)

images_faces_mrg = faces_image_sumr.merge(faces_emotions_sumr,how="left",on="image_id")
#faces_allfaces_sumr = faces_image_sumr.groupby(by='image_id').agg({"face_id":pd.Series.nunique,"face_smile_confidence":pd.Series.mean})

images_data_mrg = image_data_ss.merge(images_faces_mrg,how="left",on="image_id")

users_data_mrg = image_data_ss.merge(images_data_mrg,how="left",on="image_id")
users_data_mrg.to_csv("Glen/users_data_mrg.csv")


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
