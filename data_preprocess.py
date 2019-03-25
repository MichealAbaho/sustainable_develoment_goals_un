import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fetch_data as fd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
import json
from tabulate import tabulate


class data_preprocessing:
    def __init__(self, path):
        self.path = path
        self.file = pd.read_csv(self.path)
        # GOAL, TARGET, INDICATOR and SERIES_CODE STORED SOMEWHERE FOR FUTURE USE
        self.dataset_future = self.file[['Goal', 'Target', 'Indicator', 'SeriesCode']]

    def clean_cols(self):
        # checkout the statistics of the data
        dataset = self.file
        # cleaning up the columns
        cleaned_columns = []
        for i in dataset.columns:
            i = str(i).strip('[]')
            cleaned_columns.append(i)
        dataset.columns = cleaned_columns

        # convert the dates into one dtatype i.e string format and pick out string representing the date
        dataset['TimePeriod'] = dataset['TimePeriod'].astype(str).apply(lambda x: x[0:4])
        dataset['TimePeriod'] = pd.to_datetime(dataset['TimePeriod'], infer_datetime_format=True)
        dataset['Year'] = dataset['TimePeriod'].apply(lambda x: x.year)

        print('Number of columns before {}'.format(len(dataset.columns)))
        #
        # 'Time_Detail', 'Source', 'FootNote', 'Name of international agreement',
        #  'Reporting Type', 'Nature' are un-necessary because they contain
        #  completely irrelvant infoirmation for analysis and 'TimePeriod' was replaced by Year
        #
        a_dataset = drop_cols(dataset, ['Time_Detail', 'Source', 'FootNote', 'Name of international agreement', 'Reporting Type', 'Nature','TimePeriod'])
        print('Number of columns After {}'.format(len(a_dataset.columns)))
        #deleting last raw in the datase t becouse its empty
        a_dataset=a_dataset[:-1]

        #first action on missing values in my dataset
        a_dataset = deal_with_empty_columns(a_dataset)

        #second action of dealing with missing values
        categorical_attributes = a_dataset.dtypes[a_dataset.dtypes == object]
        for var in categorical_attributes.index:
            var = str(var)
            if (var != 'Value'):
                a_dataset[var].fillna('None', inplace=True)

        return a_dataset

    def reshaping(self):
        dataset_new = self.clean_cols()
        #remove columns not needed in our reshaped data i.e. ref self.dataset_future above
        dataset_new = dataset_new.iloc[:, 4:]

        # checkpoint 1
        # repositioning the columns to clearly view the coming changes
        first_3_vars_required = ['Year', 'SeriesDescription', 'Value']
        columns_ordered_list = [var for var in list(dataset_new.columns) if var not in first_3_vars_required]
        columns_ordered_list_changed = np.concatenate((first_3_vars_required, columns_ordered_list), axis=None)

        dataset_new_cols = list(columns_ordered_list_changed)
        dataset_new = dataset_new[dataset_new_cols]

        # sorting the dataset by Year and series description
        dataset_new = dataset_new.sort_values(by=['SeriesDescription', 'Year'], ascending=True).reset_index()
        dataset_new.drop(['index'], inplace=True, axis=1)

        dataset_new['SeriesDescription'] = label_encod(dataset_new['SeriesDescription'].astype(str))

        #Process of reshaping begins by first freezing all the columns we want as index columns
        index_cols = np.array(dataset_new.columns).tolist()
        index_cols = [col for col in index_cols if col != 'Value']
        dataset_new.set_index(index_cols, inplace=True)

        dataset_new_pivot = dataset_new.unstack('Year').reset_index()
        dataset_new_pivot['SeriesDescription'] = label_decode(dataset_new_pivot['SeriesDescription'].astype(str))

        #dataset_new.to_csv('dataset_pivot.csv')
        return dataset_new_pivot


#deleting unnesseary colmuns
def drop_cols(df, s):
    df = df.drop(s, axis=1)
    return df

#deleting empty columns
def deal_with_empty_columns(df):
    df_size = df.shape[0]
    missing_data = df.isnull().sum()
    all_null = missing_data[missing_data == df_size]
    all_null.plot.bar()
    plt.xticks(rotation=20)
    plt.show()
    df = df.drop(list(all_null.index), axis=1)
    return df

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



