#!/usr/bin/env python

"""
machine learning predictions, outputs accuracy and saves list of features.
input: x_train.npy, x_test.npy, y_train.npy, y_test.npy
output: predictions, top_feats.npy
"""

import os, sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# this script is untested, pending a working feature_selection

if __name__ =='__main__':
    np.save('data_sets/x_train.npy', x_train)
    np.save('data_sets/y_train.npy', y_train)
    np.save('data_sets/x_test.npy', x_test)
    np.save('data_sets/y_test.npy', y_test)

    model_type = sys.argv[1]

    if(model_type=='XGB'):
        model =  XGBClassifier(learning_rate=1, n_estimators=10, objective='binary:logistic', silent=True, nthreads=8)
	model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = [round(value) for value in y_pred]
	
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
