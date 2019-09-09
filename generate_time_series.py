import numpy as np
import pandas as pd
import data_preprocess as dp
import helper_functions as hf
import os
import sys
import re
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
import json
from pprint import pprint

pd.options.display.max_columns=40

class generate_time_series:
    def __init__(self, reshaped_df):
        self.reshaped_df = reshaped_df
        self.sd_time_series_dict = {}
        self.plot_dir = ''

    #returning the unique series description for a country
    def return_list_of_unique_series_des(self):
        sd_list = set([i for i in self.reshaped_df['SeriesDescription']])
        return sd_list

    def generate_combinations_per_series_des(self, sd_list, ctry):
        reshaped_data = self.reshaped_df.iloc[1:, :] #we do this to avoid fumbling with multi-indexed dataframe

        reshaped_data.columns = reshaped_data.columns.get_level_values(0)

        # rename the columns to suit your desired layout
        columns = np.array(reshaped_data.columns).tolist()

        years_columns = [i for i,j in enumerate(columns) if j == 'Value'] #obtaining indexes for all years as per reshaped df

        years = [s for s in range(2019, (2019-len(years_columns)), -1)] #obtaining the range of years in your frame given the current or latest year is 2019
        years = [years.pop() for _ in range(len(years))]

        columns = np.concatenate((columns[:years_columns[0]], years), axis=None)
        reshaped_data.columns = columns

        combinations_folder = hf.create_directories_per_series_des(name='COMBINATIONS')
        combinations_file = open(os.path.join(combinations_folder, '{} combinations.txt'.format(ctry)), 'w')
        total_list_of_combinations = []
        for sd in sd_list:
            # selecting a series description to perform time series on, selecting the most granular detail #CHECKPOINT 2
            reshaped_data_series = reshaped_data.loc[reshaped_data['SeriesDescription'] == sd]
            years_df = reshaped_data_series.iloc[:, years_columns].fillna(0)
            # You don't want to eliminate any single year irrespective of whether it's completely empty or not
            parameters = hf.fetch_parameters_for_auto_arima(reshaped_data_series.iloc[:, :-(len(years_columns))])
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

            combinations_file.write('{}\n'.format(sd))
            for i in combinations:
                combinations_file.write('{}\n'.format(str(i)))
            combinations_file.write('\n')

            combinations_df = pd.DataFrame({i:combinations})
            # print('\nAttributes to be considered in the various time series prediction include \n {}'.format(parameter_record_list))
            # print(tabulate(combinations_df, headers='keys', tablefmt=''))

            i = 0
            sd_time_series_dict = []
            for comb in combinations:
                time_serie_df = pd.DataFrame()
                time_serie_df = filled_params_df
                series_description_name = ''.join(j for j in ['{}+^'.format(str(i[1])) for i in comb if str(i[0]) != 'SeriesDescription'])

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
                    #time_serie_df = time_serie_df[time_serie_df['Year'] != 2018]
                    self.sd_time_series_dict[series_description_name] = time_serie_df
                    #sd_time_series_dict.append(time_serie_df)
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
        return  self.sd_time_series_dict, combinations, combinations_folder

if __name__=='__main__':
    data_processed = dp.data_preprocessing('allcountries.csv').reshaping()  #this line returns multiple contries
    print('\n')
    for country in data_processed:
        if country.lower() == 'belgium':
            country_dir = hf.create_directories_per_series_des('COMBINATIONS/{}'.format(country))
            data_reshaped = generate_time_series(data_processed[country])
            if data_reshaped:
                unique_sds_in_country = data_reshaped.return_list_of_unique_series_des()
                #print('The number of Unique Series_description in {}: {}'.format(country, len(unique_sds_in_country)))
                time_seies_dict, combinations, combinations_folder = data_reshaped.generate_combinations_per_series_des(unique_sds_in_country, country)
                #f = data_reshaped.sd_time_series_dict
                #print(len(time_seies_dict))
                sds, sds_list = {}, []
                goals = list(range(1, 18, 1))
                t= open('t.txt', 'w')
                for goal in goals:
                    for i,j in enumerate(time_seies_dict):
                        r = j.strip().split('+^')
                        d = [i for i in list(r) if len(i) > 0]
                        d.sort(key=lambda x:len(x))
                        t.writelines(d)
                #         try:
                #             f = [re.match(r'^\d{1,2}$', i) for i in d]
                #             f = [i for i in f if i != None]
                #             f = int(f[0].group())
                #             if f in goals and f == goal:
                #                 sds_list.append((j, time_seies_dict[j]))
                #             else:
                #                 pass
                #         except ValueError as e:
                #             print('Nothing seems to be an integer in there')
                #     sds[goal] = [i for i in sds_list]
                #     sds_list.clear()
                # for u,v in sds.items():
                #     print(u)

                # with open(os.path.join(country_dir, 'column_names.txt'), 'w') as cn:
                #     for k,v in sds.items():
                #         analysis_file = pd.DataFrame([i for i in range(2001, 2020, 1)])
                #         cn.write('{} \n'.format(str(k)))
                #         for item in v:
                #             analysis_file = pd.concat([analysis_file, item[1]['Value']], axis=1)
                #             cn.writelines(str(item[0]))
                #             cn.writelines('\n')
                #         cn.writelines('\n')
                #
                #         analysis_file_copy = analysis_file.copy()
                #         analysis_file_copy.rename(columns={analysis_file_copy.columns[0]:"Year"}, inplace=True)
                #         analysis_file_copy.set_index('Year', inplace=True)
                #         analysis_file_copy.columns = ['SD_{}'.format(i) for i in range(1, len(v)+1)]
                #
                #         analysis_file_copy.to_csv(os.path.join(country_dir, '{}_goal_{}.csv'.format(country, k)))
                #     cn.close()


