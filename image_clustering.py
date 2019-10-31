#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/Users/xuetan/Desktop/Ivey_Business Analytics/Social Media/FinalProject/Pic
Created on Tue Nov 13 16:40:52 2018

@author: xuetan
"""
#feature extraction

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

model = VGG16(weights='imagenet', include_top=False)
#model.summary()
#
img_pth = '/Users/xuetan/Desktop/Ivey_Business Analytics/Social Media/FinalProject/Pic_2/'
#
#test_path = '/Users/xuetan/Desktop/Ivey_Business Analytics/Social Media/FinalProject/Pic/402a771d5cdf2fe55ebab87796931aaa.jpg'
#img = image.load_img(test_path,target_size=(224, 224))
#img
#img_data = image.img_to_array(img)
#img_data = np.expand_dims(img_data, axis=0)
#img_data = preprocess_input(img_data)
#vgg16_feature = model.predict(img_data)
#print(vgg16_feature)

########################################
import os
list = os.listdir("/Users/xuetan/Desktop/Ivey_Business Analytics/Social Media/FinalProject/Pic_2")
#all image
list
del list[5]
list
#convert to series
list_ser = pd.Series(list)

img_list = []
for i in list:
    img_path = img_pth + i
    img_list.append(img_path)
print(len(img_list))
#####################
vgg16_feature_list = []
for i in img_list:
    img = image.load_img(i,target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    vgg16_feature_list.append(vgg16_feature_np.flatten())
print(len(vgg16_feature_list_np))
vgg16_feature_list_np = np.array(vgg16_feature_list)
kmeans = KMeans(n_clusters=3, random_state=0).fit(vgg16_feature_list_np)
y_kmeans = kmeans.predict(vgg16_feature_list_np)
y_kmeans
y_kmeans_list = y_kmeans.tolist()
#convert to series
y_kmeans_se = pd.Series(y_kmeans_list)
cluster_df = pd.DataFrame()
#add values of img_id
cluster_df["img_id"] = list_ser.values
cluster_df["cluster_no"] = y_kmeans_se.values
#write to file
cluster_df.to_csv('/Users/xuetan/Desktop/Ivey_Business Analytics/Social Media/FinalProject/cluster_df.csv')
#plot

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
X = vgg16_feature_list_np
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

#get values in each cluster
kmeans.labels_
kmeans.cluster_centers_