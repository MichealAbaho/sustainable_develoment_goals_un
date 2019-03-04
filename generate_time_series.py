import numpy as np
import pandas as pd
import data_preprocess as dp
import helper_functions as hf
import os
import sys
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt

pd.options.display.max_columns=40

class generate_time_series:
    def __init__(self, reshaped_df):
        self.reshaped_df = reshaped_df
        self.sd_time_series_dict = {}

    #returning the unique series description for a country
    def return_list_of_unique_series_des(self):
        sd_list = set([i for i in self.reshaped_df['SeriesDescription']])
        return sd_list

    #creating a directory for the plots
    def create_directories_per_series_des(self):
        plot_dir = os.path.abspath(os.path.join(os.path.curdir, 'PLOTS'))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    def generate_combinations_per_series_des(self, sd_list):
        reshaped_data = self.reshaped_df.iloc[1:, :]
        reshaped_data.columns = reshaped_data.columns.get_level_values(0)
        # rename the columns to suit your desired layout
        columns = np.array(reshaped_data.columns).tolist()
        years = [range(2000, 2019, 1)]
        columns = np.concatenate((columns[:-19], years), axis=None)
        reshaped_data.columns = columns
        #combinations_file = open('combinations.txt', 'a')
        total_list_of_combinations = []
        for sd in sd_list:
            # sd_dir = os.path.join(plot_dir, str(sd))
            # if not os.path.exists(sd_dir):
            #     os.makedirs(sd_dir)
            # selecting a series description to perform time series on, selecting the most granular detail #CHECKPOINT 2
            reshaped_data_series = reshaped_data.loc[reshaped_data['SeriesDescription'] == sd]
            years_df = reshaped_data_series.iloc[:, -19:].fillna(0)
            # You don't want to eliminate any single year irrespective of whether it's completely empty or not
            parameters = hf.fetch_parameters_for_auto_arima(reshaped_data_series.iloc[:, :-19])
            filled_params_df = reshaped_data_series[parameters]  # dataframw with attributes that are filled in

            filled_params_df = pd.concat([filled_params_df, years_df], axis=1)
            # convert all values in selected series description into floating points
            for col in filled_params_df:
                if (col in years):
                    filled_params_df[col] = list(map(float, list(filled_params_df[col])))

            # list to contain the different unique records of attributes
            parameter_record_list = []
            for i in parameters:
                unique_attribute_record = set(list(filled_params_df[i]))
                for record in unique_attribute_record:
                    attr_record_tuple = (i, record)
                    if attr_record_tuple not in parameter_record_list:
                        parameter_record_list.append(attr_record_tuple)

            combinations = hf.generate_combinations(parameter_record_list)
            total_list_of_combinations.append(len(combinations))

            #combinations_file.write('{}\n'.format(sd))
            # for i in combinations:
            #     combinations_file.write('{}\n'.format(str(i)))
            # combinations_file.write('\n')
            #
            #combinations_df = pd.DataFrame({i:combinations})
            #print('\nAttributes to be considered in the various time series prediction include \n {}'.format(parameter_record_list))
            #print(tabulate(combinations_df, headers='keys', tablefmt=''))

            i = 0
            sd_time_series_dict = []
            for comb in combinations:
                comb = sorted(comb, key=lambda x:len(x[0]), reverse=True)
                time_serie_df = pd.DataFrame()
                time_serie_df = filled_params_df
                file_saved = ''.join(j for j in ['{}_'.format(str(i[1])) for i in comb])

                for pair in comb:
                    col, val = pair
                    time_serie_df = time_serie_df[time_serie_df[col] == val]

                time_serie_df = time_serie_df.iloc[:, -19:]
                try:
                    for yr in time_serie_df:
                        time_serie_df[yr] = time_serie_df[yr].apply(lambda x: hf.convert_strings_to_floats(x))
                    time_serie_df = time_serie_df.astype(float)
                    time_serie_df = time_serie_df.stack().groupby(level=1).sum().reset_index().rename(columns={'index': 'Year', 0: 'Value'})
                    time_serie_df['Year'] = time_serie_df['Year'].apply(lambda x: datetime.strptime(x, '%Y').year)
                    time_serie_df = time_serie_df[time_serie_df['Year'] != 2018]
                    sd_time_series_dict.append(time_serie_df)
                    #print(tabulate(time_serie_df, headers='keys', tablefmt='psql'))
                    # plt.plot(range(len(time_serie_df['Year'])), time_serie_df['Value'], label=sd)
                    # plt.xlabel('year')
                    # plt.ylabel('value')
                    # plt.title(sd)
                    # plt.xticks(range(len(time_serie_df['Year'])), time_serie_df['Year'], rotation=45)
                    #plt.savefig(os.path.join(sd_dir, '{}.png'.format(file_saved)))
                    #plt.show()
                    i+=1

                except:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(exc_type, exc_obj, exc_tb.tb_lineno)

            self.sd_time_series_dict[sd] = sd_time_series_dict
        # for i in self.sd_time_series_dict:
        #     for j in (self.sd_time_series_dict[i]):
        #         print(tabulate(j, headers='keys', tablefmt='psql'))


if __name__=='__main__':
    data_processed = dp.data_preprocessing('egypt.csv')
    data_reshaped = generate_time_series(data_processed.reshaping())
    if data_reshaped:
        unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
        print('The number of Unique Series_description in Egypt: {}'.format(len(unique_sds_in_country)))
        data_reshaped.generate_combinations_per_series_des(unique_sds_in_country)
        f = data_reshaped.sd_time_series_dict
        for i in f:
            for j in (f[i]):
                print(tabulate(j, headers='keys', tablefmt='psql'))