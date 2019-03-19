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
