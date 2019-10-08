#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:29:10 2019

@author: hduser
"""


import pandas as pd
survey_raw = pd.read_pickle("survey.pickle")
image_metrics_raw = pd.read_pickle("image_metrics.pickle")
image_data_raw = pd.read_pickle("image_data.pickle")

# Build a user summary from the images_data file (data is denormalized in this file)
user_data_sumr = image_data_raw.groupby(by="user_id") \
    .agg({"user_followed_by":pd.Series.mean,"user_follows":pd.Series.mean,"image_id":pd.Series.count}) \
    .reset_index().rename(columns={"image_id":"images_count"})

# Subset columns to the ones which are useful X features
image_data_ss = image_data_raw[['image_id', 'user_id','data_memorability','image_filter','image_link']]
# Build a summary of each important feature and merge it to the images table
images_metrics_ss = image_metrics_raw[['image_id','comment_count','like_count']]

# Merge
images_mrg = image_data_ss.merge(images_metrics_ss,how="inner",on="image_id")

# now summarize images_mrg down to one row per user and that's something
# we can start some ML work on
user_images_sumr = images_mrg.groupby(by="user_id") \
    .agg(({"image_id":pd.Series.count,"data_memorability":pd.Series.mean,"comment_count":pd.Series.mean,"like_count":pd.Series.mean,"image_link":pd.Series.max})).reset_index() \
    .rename(columns={"like_count":"images_mean_like_count","comment_count":"images_mean_comment_count","memorability_score":"mean_memorability_score","image_id":"image_count","image_link":"sample_image_link"})

users_sumr = user_data_sumr.merge(user_images_sumr,how="inner",on="user_id")

users_sumr = users_sumr.add_prefix("X_")
users_sumr = users_sumr.rename(columns = {"X_user_id":"i_user_id","X_sample_image_link":"i_images_link"})
users_sumr['i_user_id']=users_sumr['i_user_id'].astype('int64')

survey_ss = survey_raw[['insta_user_id','gender','income','employed','PERMA','P','E','R','M','A','P_EMO','N_EMO']]
survey_ss.columns = survey_ss.columns.str.lower()
survey_ss = survey_ss.add_prefix("X_")
survey_ss = survey_ss.rename(columns={"X_perma":"y_perma","X_insta_user_id":"i_user_id"})
survey_ss = survey_ss.rename(columns={"X_p":"y_p" , "X_e":"y_e", "X_r":"y_r","X_m":"y_m", "X_a":"y_a","X_p_emo":"y_p_emo","X_n_emo":"y_n_emo" })


survey_user_mrg = survey_ss.merge(users_sumr, how="inner",on="i_user_id")

##########################################################################
# difficult merge incoming, this block is independant of all the code above
faces_raw = pd.read_pickle("face.pickle")
image_data_raw = pd.read_pickle("image_data.pickle")

# The faces table has multiple rows with some denormalized data.
# So - first step is to crosstabulate the emo data into individual fields
# so that the remaining data can just be deduplicated

faces_emo_ss = faces_raw[['image_id','face_id','face_emo','emo_confidence']]

faces_emo_ct = pd.crosstab(index=[faces_emo_ss.image_id, faces_emo_ss.face_id], 
                           columns = faces_emo_ss.face_emo, 
                           values=faces_emo_ss.emo_confidence, aggfunc=pd.Series.sum
                           ).reset_index()

faces_emo_ct.columns = faces_emo_ct.columns.str.lower()

faces_emo_ct.fillna(0, inplace=True)

faces_feat_dd  = faces_raw.sort_values(['image_id','face_id']).groupby(['image_id','face_id']).first().reset_index()
faces_feat_dd = faces_feat_dd.drop(['face_emo','emo_confidence'],axis=1)

faces_feat_mrg = faces_feat_dd.merge(faces_emo_ct,how='inner',on=['image_id','face_id'])

faces_feat_dummies = pd.get_dummies(faces_feat_mrg,columns=['face_gender','face_smile'])
faces_feat_dummies.columns = faces_feat_dummies.columns.str.lower()

faces_feat_sumr = faces_feat_dummies.groupby(['image_id']) \
    .agg({"face_id":pd.Series.count,"face_gender_male":pd.Series.sum, \
          "face_gender_female":pd.Series.sum,"face_age_range_high":pd.Series.mean, \
          "face_age_range_low":pd.Series.mean,"face_smile_true":pd.Series.sum, \
          "face_smile_false":pd.Series.sum,"angry":pd.Series.mean,"happy":pd.Series.mean, \
          "calm":pd.Series.mean, "confused":pd.Series.mean, "disgusted":pd.Series.mean, \
          "sad":pd.Series.mean}).reset_index() 


faces_feat_sumr = faces_feat_sumr.rename(columns={"face_id":"face_count","face_gender_male":"male_count", \
             "face_gender_female":"female_count","face_smile_true":"smile_count", \
             "face_smile_false":"frown_count"})

faces_feat_user = faces_feat_sumr.merge(image_data_raw[["image_id","user_id"]],how="inner")
faces_feat_tots= faces_feat_user.groupby("user_id") \
    .agg({"image_id":pd.Series.count,"face_count":pd.Series.mean, \
          "male_count":pd.Series.mean, "female_count":pd.Series.mean, \
          "smile_count":pd.Series.mean, "frown_count":pd.Series.mean, "angry":pd.Series.mean, \
          "happy":pd.Series.mean,"calm":pd.Series.mean,"confused":pd.Series.mean, "disgusted":pd.Series.mean, \
          "sad":pd.Series.mean}).reset_index()

faces_feat_pretty = faces_feat_tots.rename(columns={"image_id":"images_w_faces_count","face_count":"img_mean_face_count", \
    "male_count":"img_mean_male_count","female_count":"img_mean_female_count", "smile_count":"img_mean_smile_count", \
    "frown_count":"img_mean_frown_count","sad":"img_avg_sad_score","angry":"img_avg_angry_score","happy":"img_avg_happy_score", \
    "calm":"img_avg_calm_score","confused":"img_avg_confused_score","disgusted":"img_avg_disgusted_score"})

faces_feat_pretty = faces_feat_pretty.add_prefix("X_")
faces_feat_pretty.rename(columns={"X_user_id":"i_user_id"},inplace=True)
faces_feat_pretty['i_user_id'] = faces_feat_pretty['i_user_id'].astype("int64")

survey_user_face_mrg = survey_user_mrg.merge(faces_feat_pretty,on="i_user_id")

cols = survey_user_face_mrg.columns.tolist()

cols.sort(reverse=True)

survey_user_face_mrg = survey_user_face_mrg[cols]

survey_user_mrg.to_pickle("survey_user_mrg.pickle")
