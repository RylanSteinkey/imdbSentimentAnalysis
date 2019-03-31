#!/usr/bin/env python

"""
This script loads the data.csv, splits the text into features, and saves it to a pandas dataframe
input: data.csv
output: words.pd
ex.
       | love | good | terrible |
|3118_1|   0  |   0  |    2     |
|  .   |   .  |   .  |    .     |
|  .   |   .  |   .  |    .     |
|3200_8|   2  |   1  |    0     |
"""

# things to do:
# build array of all words seen
# make numpy array of zeroes
# read each review and change the count for each review for each word seen

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

def find_index(arr, element):
    for i, val in enumerate(arr):
        if(val == element):
            return i

if __name__ == "__main__":

    data_df = pd.read_csv('data.csv')

    all_words = set(['the'])
    common = set(stopwords.words('english'))

    for review in data_df.values:
        # Remove all non alphanumeric charecters
        edited_review = re.sub(r'[^\w]', ' ', review[4])

        # lowercase all charecters
        edited_review = [i.lower() for i in edited_review.split()]

        # removed common words (try removing this to see accuracy change later)
        edited_review = filter(lambda i: not i in common, edited_review)

        # add this reviews words to our master list of words (all_words)
        all_words = set(edited_review).union(all_words)

    # At this point all_words should be a set of all words seen
    num_words = len(all_words)

    words_matrix = np.zeros((1000,num_words), dtype='uint8')

    for j, review in enumerate(data_df.values):
        # Remove all non alphanumeric charecters
        edited_review = re.sub(r'[^\w]', ' ', review[4])

        # lowercase all charecters
        edited_review = [i.lower() for i in edited_review.split()]

        # removed common words (try removing this to see accuracy change later)
        edited_review = filter(lambda i: not i in common, edited_review)

        for word in edited_review:
            words_matrix[j][find_index(all_words, word)] += 1

    words_df = pd.DataFrame(data = words_matrix, index = data_df.values[:,1], columns = all_words)

    words_df.to_pickle('words.df')
