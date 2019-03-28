#!/usr/bin/env python

"""
This script loads the feature dataframe, the classifications, and preforms feature selection and splits the data.
input: words.pd, data.csv
output: x_train.npy, x_test.npy, y_train.npy, y_test.npy
"""

# things to do:
# for train and test, run selectKbest scikit feature selection
# split data into data (x_train) and the pos/neg classification (y_train)

import sys, os
import numpy as np
import pandas as pd

num_feats = 1000

words_df = pd.read_pickle('words.df')
data_df = pd.read_csv('data.csv')

all_file_names = data_df['file_name']

# split file names into positive and negative
positive = np.hstack((all_file_names[0:12500],all_file_names[25000:37500]))
#negative = np.hstack((all_file_names[12500:25000],all_file_names[37500:50000]))

# split file names into train and test
train = np.hstack((all_file_names[0:25000]))

train_mask = np.zeros(50000)
pos_mask = np.zeros(50000)
 
# go through each sample in words_df and label which ones are training
for i, review in enumerate(words_df.index.tolist()):
    if review in train:
        train_mask[i]=1

# split data into train and test
train_mask = [i==1 for i in train_mask]

train_data = words_df.values[train_mask]
train_names = all_file_names[train_mask]

test_mask = [i=='false' for i in train_mask]

test_data = words_df.values[test_mask]
test_names = all_file_names[test_mask]

# so now we have the test data (x_test) and training data (x_train) we need the positive/negative labels

print(len(train_names))
"""
THIS IS WHERE THE SCRIPT FAILS
train names should be 25000 elements long but its actually ~32,000
causes index error on next loop
"""
sys.exit()
y_train = np.zeros(25000)
for i, name in enumerate(train_names):
    if name in positive:
        y_train[i]=1
y_test = np.zeros(25000)
for i, name in enumerate(test_names):
    if name in positive:
        y_test[i]=1

x_test = test_data
x_train = train_data

# now we have x_train, x_test, y_train, y_test we can do feature selection

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

# first we make a feature selection object
feat_select_object = SelectKBest(f_classif, k=num_feats)

# now we learn which features are best, while removing unimportant once
x_train = feat_select_object.fit_transform(x_train, y_train)

# now we reduce the x_test to the same features as x_train, note that the feat_select_object never saw the testing set when choosing what features to keep
x_test = feat_select_object.transform(x_test)

# now save everything

np.save('data_sets/x_train.npy', x_train)
np.save('data_sets/y_train.npy', y_train)
np.save('data_sets/x_test.npy', x_test)
np.save('data_sets/y_test.npy', y_test)
