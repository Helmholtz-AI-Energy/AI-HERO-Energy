import os
import numpy as np
import pandas as pd
from datetime import datetime
import glob

# This directory should only contain the raw entso-e csv files
home = '/home/arvid/'
data_path = home + 'Data/entso-e/raw/'

# output directory
save_path = home + 'Data/entso-e/'
dst = ['2015-03-29 03:00:00+00:00', '2016-03-27 03:00:00+00:00', '2017-03-26 03:00:00+00:00',
       '2018-03-25 03:00:00+00:00', '2019-03-31 03:00:00+00:00', '2020-03-29 03:00:00+00:00',
       '2015-10-25 03:00:00+00:00', '2016-10-30 03:00:00+00:00', '2017-10-29 03:00:00+00:00',
       '2018-10-28 03:00:00+00:00', '2019-10-27 03:00:00+00:00', '2020-10-25 03:00:00+00:00']

# For more stable measurement the training set is used [extension_factor] times for eval consumption measurement
extension_factor = 5


def prep_dataframe(file, country):
    raw_data = pd.read_csv(file, delimiter=',')
    raw_data['year'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z").year for time in raw_data['start']]
    raw_data['posix_start'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z").timestamp() for time in raw_data['start']]
    raw_data['posix_stop'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z").timestamp() for time in raw_data['end']]
    raw_data['date'] = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z").date() for time in raw_data['start']]
    raw_data['deltaT'] = raw_data['posix_stop'] - raw_data['posix_start']
    raw_data['time_diff'] = raw_data['posix_start'].diff()
    timestep = raw_data['time_diff'].median()
    step = int(3600 / timestep)
    delta = int(86400 / timestep)

    # Drop everything >2020
    raw_data = raw_data.drop(raw_data.loc[raw_data['year'] == 2020].index)

    # Replace 0.0
    zeroes = raw_data.loc[raw_data['load'] == 0.0].index
    for z in zeroes:
        stack = []
        c = 1
        while (z - delta * c) >= 0 and c < 5:
            stack.append(raw_data.loc[z - c * delta, 'load'])
            c += 1
        raw_data.loc[z, 'load'] = np.mean(stack)

    # Replace NaNs
    nans = raw_data.loc[np.isnan(raw_data['load'])].index
    for n in nans:
        stack = []
        stack.append(raw_data.loc[0, 'load'])
        c = 1
        while (n - delta * c) >= 0 and c < 5:
            stack.append(raw_data.loc[n - c * delta, 'load'])
            c += 1
        raw_data.loc[n, 'load'] = np.mean(stack)

    # Fill Missing TimeSteps
    missing = raw_data.loc[raw_data['time_diff'] != timestep].index
    for idx in missing:
        if idx == 0:
            continue

        if raw_data.loc[idx, 'start'] not in dst:
            gap = int((raw_data.loc[idx, 'posix_stop'] - raw_data.loc[idx - 1, 'posix_stop']) / timestep - raw_data.loc[
                idx, 'deltaT'] / timestep)
            for g in range(gap):
                delta = int(86400 / timestep)
                stack = []
                c = 1
                while ((idx - 1 + g) - delta * c) >= 0 and c < 5:
                    stack.append(raw_data.loc[(idx - 1 + g) - c * delta, 'load'])
                    c += 1

                l = np.mean(stack)
                row = pd.DataFrame({'start': ['X'], 'end': ['Y'], 'load': [l], 'year': [raw_data.loc[idx - 1, 'year']],
                                    'posix_start': [raw_data.loc[idx - 1, 'posix_start'] + timestep * (g + 1)],
                                    'posix_stop': [raw_data.loc[idx - 1, 'posix_start'] + timestep * (g + 2)],
                                    'deltaT': [timestep], 'time_diff': [timestep]})
                raw_data = pd.concat([raw_data, row])

            # raw_data=raw_data.drop([0], axis=1)
    raw_data = raw_data.sort_values('posix_start')
    raw_data = raw_data.reset_index(drop=True)

    # Set new time and drop useless columns
    train = pd.DataFrame()
    valid = pd.DataFrame()
    test = pd.DataFrame()
    idx = raw_data.index % step
    reduced = raw_data.iloc[idx == 0].reset_index(drop=True)
    integral = raw_data.rolling(step).sum().iloc[idx == step - 1].reset_index(drop=True)
    if reduced.shape[0] != integral.shape[0]:
        print("Error: Integral and Reduced do not match")
        return 0

    # time = raw_data['posix_start'].iloc[idx == 0].reset_index(drop=True)
    train_set = reduced.loc[reduced['year'] <= 2017].index
    valid_set = integral.loc[reduced['year'] == 2018].index
    test_set = integral.loc[reduced['year'] == 2019].index

    train['Load [MW]'] = integral['load'][train_set]
    valid['Load [MW]'] = integral['load'][valid_set]
    test['Load [MW]'] = integral['load'][test_set]

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train['Time [s]'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in
                         reduced['posix_start'][train_set]]
    valid['Time [s]'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in
                         reduced['posix_start'][valid_set]]
    test['Time [s]'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in
                        reduced['posix_start'][test_set]]

    train['Country'] = country
    valid['Country'] = country
    test['Country'] = country

    return train, valid, test


def main():
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    energy_test_data = pd.DataFrame()

    for file in glob.glob(data_path + "*.csv"):
        country = (file.split('.')[0]).split('/')[-1]
        print(country)
        if country in ['lu']:
            continue
        train, valid, test = prep_dataframe(file, country)

        train_data = pd.concat([train_data, train], ignore_index=True)
        valid_data = pd.concat([valid_data, valid], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)
        for n in range(extension_factor):
            energy_test_data = pd.concat([energy_test_data, train], ignore_index=True)

    test_data = test_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    energy_test_data = energy_test_data.reset_index(drop=True)

    train_data.to_csv(save_path + 'train.csv',  header=True, index=False)
    valid_data.to_csv(save_path + 'valid.csv',  header=True, index=False)
    test_data.to_csv(save_path + 'test.csv',  header=True, index=False)
    energy_test_data.to_csv(save_path + 'energy_test.csv',  header=True, index=False)


if __name__ == '__main__':
    print('Running data preparation')
    main()
