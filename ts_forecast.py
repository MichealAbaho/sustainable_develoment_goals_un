import numpy as np
from datetime import datetime
import pandas as pd
import generate_time_series as gts
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
import math
import data_preprocess as dp
from tabulate import tabulate
import helper_functions as hf
from statsmodels.tsa.arima_model import AR, ARIMA, ARMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from fbprophet import Prophet
import json
import warnings
import sys
import os
import warnings
import multiprocessing as mlp
from tqdm import *
import itertools as it

warnings.filterwarnings("ignore", category=Warning)
#speeding up our timeseries
from dask.distributed import Client, LocalCluster
import dask
import time

#set the prediction period before hand
years_to_predict = range(2000, 2031, 1)

pool = mlp.Pool(mlp.cpu_count())

#append all the time series for each country keeping a copy of both the series to use during the training as well as the one for validation
def build_series(dict_):
    n = 4
    training_series = []
    validation_series = []

    for sd in dict_:
        df = dict_[sd]
        if len(df) == 19:
            v = df[df['Value'] == 0]
            #ensure you have atleast 4 values are not null values
            if (len(v) <= 15):
                #change the date column to a date type
                df['Year'] = df['Year'].astype(str).apply(lambda x: hf.obtain_date(x))
                #df.set_index('Year', inplace=True)

                #UNDO ONLY WHEN USING FACEBOOKPROPHET
                df.columns = ['ds','y2']
                y = [np.NaN if i==0 else i for i in df.y2]
                df.drop('y2', inplace=True, axis=1)
                df['y'] = y

                train, validate = df[:(len(df) - n)], df[(len(df) - n):]
                actual = validate.y.tolist()
                #append all the time series for each country keeping a copy of both the series to use during the training as well as the one for validation
                training_series.append((sd, df['y'], train))
                validation_series.append(actual)

    return training_series, validation_series

def tS_forecast(train, validate):
    list_of_country_time_series = []
    # FORECAST
    try:
        # FACEBOOK PROHPET
        train_set = [i[2] for i in train]
        sds = [i[0] for i in train]
        years = [i[1] for i in train]
        predictions = list(tqdm(pool.imap(run_fb, train_set), total=len(train_set)))
        if (len(sds) == len(predictions) == len(validate)):
            json_dict = {}
            for sd,ts,vd,y in zip(sds, predictions, validate, predictions):
                r_mse, mape = evaluate_facebook_prophet(ts, vd)
                json_dict['Series_description'] = sd
                json_dict['Predicted_period'] = list(years_to_predict)[-13:]
                json_dict['FB_prophet'] = list(ts['yhat'])[-13:]
                json_dict['RMSE'] = r_mse
                json_dict['MAPE'] = mape
                json_dict['org_values'] = list(y)

                if json_dict:
                    list_of_country_time_series.append(json_dict)
                json_dict = {}
        else:
            'Raise Eyebrows, Something is Wrong and its because of the below \n ' \
            'length of training set-{} is not equal to the length of validation set-{} or legnth of the series description list-{}'.format(len(predictions),
                                                                                                                                           len(validate),                                                                                                                                    len(sds))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type)
        print(exc_obj)
        print(exc_tb.tb_lineno)

    return list_of_country_time_series


#@dask.delayed
def run_fb(train):
    fb = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
    fb_model = fb.fit(train)
    future = fb_model.make_future_dataframe(periods=len(years_to_predict))
    fb_predict = fb.predict(future)
    return fb_predict

def evaluate_facebook_prophet(fb_predict, validate):
    predictions = list(fb_predict['yhat'])[:18][-4:]
    actual = validate.y.tolist()
    r_mse, mape = prediction_accuracies(predictions[:4], actual)
    return r_mse, mape


#take a close look at rolling stats
def examine_rolling_metrics(df):
    rol_mean = df.rolling(2).mean()
    rol_std = df.rolling(2).std()

    # Plot rolling statistics:
    orig = plt.plot(df, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

#examine the stationarity of the time series
def exmine_the_stationarity(df, sd, file, send_p_values=False):
    result = adfuller(df.Value.dropna())
    if send_p_values == True:
        comb_file = open(file, 'a')
        try:
            comb_file.write('\n{}'.format(sd))
            comb_file.write('\nADF Statistic: {:.4f}'.format(result[0]))
            comb_file.write('\np-value: {:.4f}'.format(result[1]))
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])

        except Exception as e:
            print(sd)
            print(e)
        comb_file.close()
    return result[1]

#p,d,q exmination
def find_d(sd, df):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.Value);
    axes[0, 0].set_title('Original')
    plt.title('{}'.format(sd))
    plot_acf(df.Value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.Value.diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df.Value.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df.Value.diff().diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df.Value.diff().diff().dropna(), ax=axes[2, 1])
    plt.show()

def find_p(sd, df):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df.Value);
    axes[0, 0].set_title('Original')
    plt.title('{}'.format(sd))
    plot_pacf(df.Value, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df.Value.diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_pacf(df.Value.diff().dropna(), ax=axes[1, 1])

    plt.show()

def validity_of_determined_p_d_q(sd, df):
    try:
        model = ARIMA(df, order=(0, 1, 1))
        model_fit = model.fit(disp=0)
        model_fit.plot_predict(dynamic=False)
        plt.xticks(list(df.index))
        plt.show()

    except Exception as e:
        print(sd)
        print(e)

def prediction_accuracies(prediction, actual):
    difference = np.subtract(prediction, actual)
    mean_abs_err = np.nanmean(np.abs(difference))
    rmse = np.nanmean(difference**2)**.5
    return rmse, np.mean(difference)


if __name__=='__main__':
    cluster = LocalCluster(n_workers=5, scheduler_port=0, diagnostics_port=0, processes=False)
    client = Client(cluster)
    initial_time = time.time()
    t = []

    #data_processed = dp.data_preprocessing('allcountries.csv').reshaping()
    countriesdf = pd.read_csv('3countries.csv')
    clean_data = dask.delayed(dp.data_preprocessing)('3countries.csv')
    data_processed = dask.delayed(clean_data.reshaping())
    data_processed = dask.delayed(data_processed).compute()

    for country in data_processed:
        print(country)
        country_dir = hf.create_directories_per_series_des(name=country)
        with open(os.path.join(country_dir, 'sdgs_time_series_fb.json'), 'w') as sdgs:
            with open(os.path.join(country_dir, 'combinations.txt'), 'w') as comb_file:
                data_reshaped = gts.generate_time_series(data_processed[country])
                if data_reshaped:
                    unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
                    print('The number of Unique Series_description in {}: {}'.format(country, len(unique_sds_in_country)))
                    country_time_series, combinations = data_reshaped.generate_combinations_per_series_des(unique_sds_in_country)

                    print("Started ###########################################################")
                    start_time = time.time()
                    training_series, validation_series = build_series(country_time_series)

                    country_time_series_list = tS_forecast(training_series, validation_series)
                    print('Forecasting for {} Completed'.format(country))
                    end_time = time.time()
                    print('EXECUTION TIME ############################################################# \n{}'.format((end_time - start_time)))
                    json.dump(country_time_series_list, sdgs, indent=2, sort_keys=True)

                    comb_file.write('{}\n'.format(country))
                    for i in combinations:
                        comb_file.write('{}\n'.format(str(i)))
                    comb_file.write('\n')

                comb_file.close()
            sdgs.close()

    final_time = time.time()
    print('EXECUTION TIME ############################################################# \n{}'.format((initial_time-final_time)))

