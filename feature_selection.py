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

words_matrix = words_df.values

x_train = words_matrix[0:500]  # first 500 are train
x_test = words_matrix[500:1000] # last  500 are test

y_train = np.zeros((500))
y_train[250:500] = 1
y_test = y_train

assert(np.sum(y_train)==250 and np.sum(y_test)==250)

# now we have x_train, x_test, y_train, y_test we can do feature selection

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif


feats = np.asarray(list(words_df.columns.values)).reshape(1,-1)

# first we make a feature selection object
feat_select_object = SelectKBest(f_classif, k=num_feats)

# now we learn which features are best, while removing unimportant once
x_train = feat_select_object.fit_transform(x_train, y_train)

# now we reduce the x_test to the same features as x_train, note that the feat_select_object never saw the testing set when choosing what features to keep
x_test = feat_select_object.transform(x_test)

# we also want to keep track of which words the features are
feats = feat_select_object.transform(feats)

# now save everything

np.save('data_sets/x_train.npy', x_train)
np.save('data_sets/y_train.npy', y_train)
np.save('data_sets/x_test.npy', x_test)
np.save('data_sets/y_test.npy', y_test)

np.save('data_sets/feats.npy', feats[0])
