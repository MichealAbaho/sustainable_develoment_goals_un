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

warnings.filterwarnings("ignore", category=Warning)
#speeding up our timeseries
from dask.distributed import Client, LocalCluster
import dask
import time

def tS_forecast(dict_):
    n = 4
    list_of_country_time_series = []
    json_dict = {}
    for sd in dict_:
        df = dict_[sd]
        if len(df) == 19:
            v = df[df['Value'] == 0]
            #ensure you have atleast 4 values are not null values
            if (len(v) <= 15):
                years_to_predict = range(2000, 2031, 1)
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
                #FORECAST
                try:
                   #FACEBOOK PROHPET
                    fb_predict, r_mse, mape = fb(train, validate, years_to_predict)
                    json_dict['Series_description'] = sd
                    json_dict['Predicted_period'] = list(years_to_predict)[-13:]
                    json_dict['FB_prophet'] = list(fb_predict['yhat'])[-13:]
                    json_dict['RMSE'] = r_mse
                    json_dict['MAPE'] = mape
                    json_dict['org_values'] = list(df['y'])

                    list_of_country_time_series.append(json_dict)
                    json_dict = {}

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(exc_type)
                    print(exc_obj)
                    print(exc_tb.tb_lineno)

    return list_of_country_time_series

# @dask.delayed
def fb(train, validate, years_to_predict):
    fb = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False,uncertainty_samples=100)
    fb_model = fb.fit(train)
    future = fb_model.make_future_dataframe(periods=len(years_to_predict))
    fb_predict = fb.predict(future)
    predictions = list(fb_predict['yhat'])[:18][-4:]
    actual = validate.y.tolist()
    r_mse, mape = prediction_accuracies(predictions[:4], actual)
    return fb_predict, r_mse, mape



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
    start_time = time.time()
    t = []
    print("Started ###########################################################")
    data_processed = dp.data_preprocessing('data.csv').reshaping()
    #clean_data = dask.delayed(dp.data_preprocessing)('3countries.csv')
    # data_processed = dask.delayed(clean_data.reshaping())
    # data_processed = dask.delayed(data_processed).compute()

    for country in data_processed:
        country_dir = hf.create_directories_per_series_des(name=country)
        with open(os.path.join(country_dir, 'sdgs_time_series_fb.json'), 'w') as sdgs:
            with open(os.path.join(country_dir, 'combinations.txt'), 'w') as comb_file:
                data_reshaped = gts.generate_time_series(data_processed[country])
                if data_reshaped:
                    unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
                    print('The number of Unique Series_description in {}: {}'.format(country, len(unique_sds_in_country)))
                    country_time_series, combinations = data_reshaped.generate_combinations_per_series_des(unique_sds_in_country)

                    country_time_series_list = tS_forecast(country_time_series)

                    json.dump(country_time_series_list, sdgs, indent=2, sort_keys=True)

                    comb_file.write('{}\n'.format(country))
                    for i in combinations:
                        comb_file.write('{}\n'.format(str(i)))
                    comb_file.write('\n')

                comb_file.close()
            sdgs.close()

    print('{} Complete'.format(country))
    end_time = time.time()
    print('EXECUTION TIME ############################################################# \n{}'.format((end_time-start_time)))

