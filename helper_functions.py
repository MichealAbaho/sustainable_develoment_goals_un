import itertools
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from collections import Counter
import string
from datetime import datetime

import numpy as np
import json

# a generator object to return multiple combinations
def generate_combinations(x):
    f, d = [], []
    attrs = list(set([i[0] for i in x]))
    for i in attrs:
        for j in x:
            if (j[0] == i):
                f.append((i, j[1]))
        g = f.copy()
        d.append(g)
        f.clear()

    combinations = list((itertools.product(*d)))
    return combinations

# returns columns in a dataframe that atleast have a single filled value and not completely empty with Nan's
def fetch_parameters_for_auto_arima(data):
    data = data.replace('None', np.NaN)
    attr_parameters = [attr for attr in data if (data[attr].isnull().sum() != len(data[attr]))]
    return attr_parameters


def re_scale(elem):
    min_max = MinMaxScaler(feature_range=(0, 1))
    return min_max.fit_transform(elem)

#obtaining datetime object for the date in the time series
def obtain_date(x):
    x = datetime.strptime(x, '%Y')
    return x

# tansforming series description by ordering the duplicate series giving them a count from 0 to n (n-total number of duplicates per seriesdescription)
def label_encod(elem):
    unique_series = list(Counter(elem).keys())
    unique_series = [str(i) for i in unique_series]
    encoded_list = []
    for unique_serie in unique_series:
        duplicate_series = [i for i in elem if i == unique_serie]
        duplicate_series_transformed = [str(i) + '__' + str(j) for j, i in enumerate(duplicate_series)]
        for i in duplicate_series_transformed:
            encoded_list.append(i)
        duplicate_series.clear()
    return encoded_list


def label_decode(elem):
    decoded_list = []
    for item in elem:
        split_item = item.split('__')
        decoded_list.append(split_item[0])
    return decoded_list


def one_encod(elem):
    o_encod = OneHotEncoder()
    return o_encod.fit_transform(elem)

def convert_strings_to_floats(x):
    if (type(x) == str):
        if not (x.__contains__('>')):
            return float(x.replace(',', ''))
        else:
            return x
    elif (type(x) == float or type(x) == int):
        return float(x)
    else:
        return x


def modelling(x, y):
    X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2)
    rf = RandomForestRegressor()
    rf_model = rf.fit(X_train, Y_train)
    y_pred = rf_model.predict(x_test)
    rmse = np.sqrt(-cross_val_score(rf_model, X_train, Y_train, scoring='neg_mean_squared_error', cv=10))
    prediction_error = mean_squared_error(y_test, y_pred)
    return rf_model, y_pred, prediction_error