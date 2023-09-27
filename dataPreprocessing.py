import numpy as np
import pandas as pd
import re
import pickle

# import tensorflow as tf
# from sklearn.model_selection import train_test_split

'''Dataset Dictionary'''
data_dir = './Dataset/ml-1m/'

'''Read the data'''
# Extract from README
moviesColumn = ['movieId', 'title', 'genres']
ratingsColumn = ['userId', 'movieId', 'ratings', 'timestamps']
usersColumn = ['userId', 'sex', 'age', 'occupation', 'zipCode']

# Use pandas to read the data with separation "::"
movieData = pd.read_table((data_dir + 'movies.dat'), sep='::', names=moviesColumn, encoding="ISO-8859-1", engine='python')
ratingsData = pd.read_table((data_dir + 'ratings.dat'), sep='::', names=ratingsColumn, encoding="ISO-8859-1", engine='python')
usersData = pd.read_table((data_dir + 'users.dat'), sep='::', names=usersColumn, encoding="ISO-8859-1", engine='python')

'''Data Clean: Keep the useful Data'''
# Preprocessing the user data
usersData = usersData.drop(['zipCode'], axis=1)
userBackup = usersData.values
sex_dict = {'F': 0, 'M': 1}
usersData['sex'] = usersData['sex'].map(sex_dict)
age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
# age_map = {val: ii for ii, val in enumerate(set(usersData['age']))}
usersData['age'] = usersData['age'].map(age_map)

# Preprocessing the movie data
movieBackup = movieData.values
pattern = re.compile(r'^(.*)\((\d+)\)$')  # Remove the years in the label
titleSet = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movieData['title']))}
movieData['title'] = movieData['title'].map(titleSet)

# modify the genres data
genresDict = set()
for val in movieData['genres'].str.split('|'):
    genresDict.update(val)

genresDict.add('<PAD>')
# transfer the genres into list
genres2int = {val: ii for ii, val in enumerate(genresDict)}
genresMap = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movieData['genres']))}
# translate the genres name into the same length
for key in genresMap:
    for cnt in range(max(genres2int.values()) - len(genresMap[key])):
        genresMap[key].insert(len(genresMap[key]) + cnt, genres2int['<PAD>'])
movieData['genres'] = movieData['genres'].map(genresMap)

# translate the title of the movie into dictionary
titleDict = set()
for val in movieData['title'].str.split(): titleDict.update(val)
titleDict.add('<PAD>')
title2int = {val: ii for ii, val in enumerate(titleDict)}
titleLength = 15
titleMap = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movieData['title']))}
for key in titleMap:
    for cnt in range(titleLength - len(titleMap[key])):
        titleMap[key].insert(len(titleMap[key]) + cnt, title2int['<PAD>'])

movieData['title'] = movieData['title'].map(titleMap)

# Preprocessing the ratings data
ratingsData.drop(['timestamps'],axis=1, inplace=True)

# Merge three dataset into one
dataset = pd.merge(pd.merge(ratingsData, usersData), movieData)

# split the result dataset(ratings) and features dataset(titles, genres, occupations,...)
featuresVal, resVal = dataset.drop(['ratings'], axis=1).values, dataset['ratings'].values


# save the processed dataset as the local file
with open('./Dataset/processedData/preprocess.pkl', 'wb') as f:
    pickle.dump((titleLength,
                 titleMap,
                 genres2int,
                 featuresVal,
                 resVal,
                 ratingsData,
                 usersData,
                 movieData,
                 dataset,
                 movieBackup,
                 userBackup),
                f)
print('Data Process Done')
