#!/usr/bin/env python

"""
machine learning predictions, outputs accuracy and saves list of features.
input: x_train.npy, x_test.npy, y_train.npy, y_test.npy
output: predictions, top_feats.npy
"""

import os, sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# this script is untested, pending a working feature_selection

if __name__ =='__main__':
    # x_train is the training data (words counts)
    x_train = np.load('data_sets/x_train.npy')
    # y_train is the pos/neg labelling of the x_train
    y_train = np.load('data_sets/y_train.npy')
    x_test  = np.load('data_sets/x_test.npy')
    y_test  = np.load('data_sets/y_test.npy')

    model_type = sys.argv[1]

    if(model_type == 'XGB'):
        # declare model
        model =  XGBClassifier(learning_rate=1, n_estimators=10, objective='binary:logistic', silent=True, nthreads=8)
        # train model
        model.fit(x_train, y_train)
        # make prediction on testing data
        y_pred = model.predict(x_test)
        # round to nearest class (xgboost thing)
        y_pred = [round(value) for value in y_pred]
        # compare predictions to actual classes (y_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # now get importance of each word
        importances = model.feature_importances_
        feats = np.load('data_sets/feats.npy')

        for i in range(len(importances)):
            top_index = np.argmax(importances)
            if(importances[top_index]==0):
                break
            print(feats[top_index], importances[top_index])
            importances[top_index] = 0

    if(model_type == 'SVM'):
        from sklearn import svm
        model = svm.SVC()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

    if(model_type == 'MNB'):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

    if(model_type== 'ANN'):
        from keras.layers.core import Dense, Dropout, Activation
        from keras.models import Sequential
        from keras.callbacks import EarlyStopping

        num_feats = len(x_train[0])
        model = Sequential()

        early_stop = EarlyStopping(monitor='loss', patience=0, verbose=1, min_delta=0.005, mode='auto')

        model.add(Dense(num_feats,activation='relu',input_dim=(num_feats)))
        model.add(Dropout(0.50))
        model.add(Dense(500, activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.50))
        model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        model.fit(x_train, y_train, epochs=25, verbose=1, callbacks = [early_stop])

        y_pred = model.predict_classes(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
