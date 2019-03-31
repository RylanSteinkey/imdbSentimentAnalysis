#!/usr/bin/env python

"""
This script loads the raw data into a csv for easier reading
input: /aclImdb directory
output: data.csv
"""

import numpy as np
import pandas as pd
import os, sys

if __name__ == "__main__":

    data = np.ndarray((100,4),dtype='object')

    counter = 0
    for split in ['train', 'test']:
        for sentiment in ['neg', 'pos']:
            path  =  "aclImdb/"+split+"/"+sentiment
            for file in os.listdir(path):
                #print(sentiment, split, file)
                text_file = open(path+'/'+file, 'r')

                data[counter][0] = file.split('.')[0]
                data[counter][1] = split
                data[counter][2] = sentiment
                data[counter][3] = text_file.readlines()[0]
                counter+=1

    df = pd.DataFrame(data = data, columns = ['file_name','split','sentiment','text'])
    df.to_csv('data.csv')
