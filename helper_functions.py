import itertools
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from collections import Counter
import string
from datetime import datetime
import pandas as pd
import pprint
import os
from glob import glob
import numpy as np
import json
from tabulate import tabulate
import re

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

#encode the series description names in
def query_for_series_code(org_file):

    #look through current director for all sub-directories assuming every sub-directory belongs to a country
    current_country_dirs = list(dir for dir in os.listdir('../SDGS/') if os.path.isdir(dir))
    #escape sub-directories the likes of '.git', '__pycache__' and many others
    current_country_dirs = [i for i in current_country_dirs if not re.search(r'\.|\_', i, re.IGNORECASE)]
    #open each sub-directory
    for i, country in enumerate(current_country_dirs):
        new_data = []
        #pick only json files, because they're all you want to encode
        json_fname = glob(os.path.join(os.path.abspath(country), '*.json'))
        if json_fname:
            json_file_list = [i for i in json_fname if i.__contains__('series')]
            json_file = json_file_list[0]
            json_dir_file = os.path.dirname(json_file)
            with open(json_file, 'r') as rb:
                org = pd.read_csv(org_file, low_memory=False)
                country_org = org[org['GeoAreaName'] == country]
                org_of_interest = country_org[['Goal', 'Target', 'Indicator', 'SeriesCode', 'SeriesDescription']]
                load_file = json.load(rb)
                q = 0
                for m in load_file:
                    q+=1
                    for a,b,c,x,y in zip(org_of_interest['Goal'], org_of_interest['Target'], org_of_interest['Indicator'], org_of_interest['SeriesCode'], org_of_interest['SeriesDescription']):
                        if str(m['Series_description']).__contains__(str(y)):
                            s = m['Series_description'].split('_')
                            s = sorted(s, key=lambda x: len(x), reverse=True)

                            w = [a,b,c]
                            for d in w:
                                if d in s:
                                    s.remove(d)

                            for o in range(len(s)):
                                for t in [w.pop() for _ in range(len(w))]:
                                    if str(t).strip()[-2:] == '.0':
                                        t = str(int(t))
                                    else:
                                        t = str(t)
                                    s.insert(o+1, t)
                                    #print(s)
                                break

                            e = '_'.join(i for i in s).rstrip('_')
                            m['Series_description'] = e.replace(str(y), x)
                        else:
                            pass

                    new_data.append(m)

            with open(os.path.join(json_dir_file, 'encoded_TS_fb.json'), 'w') as fb:
                json.dump(new_data, fb, indent=2, sort_keys=True)
            fb.close()
            rb.close()

#query_for_series_code('allcountries.csv')

#creating a directory for the plots
def create_directories_per_series_des(name=''):
    _dir = os.path.abspath(os.path.join(os.path.curdir, name))
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir

def modelling(x, y):
    X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2)
    rf = RandomForestRegressor()
    rf_model = rf.fit(X_train, Y_train)
    y_pred = rf_model.predict(x_test)
    rmse = np.sqrt(-cross_val_score(rf_model, X_train, Y_train, scoring='neg_mean_squared_error', cv=10))
    prediction_error = mean_squared_error(y_test, y_pred)
    return rf_model, y_pred, prediction_error