import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from copy import deepcopy
from matplotlib import pyplot as plt

districts = {"City Of Tshwane Metro": 3522325,
            "City Of Johannesburg Metro": 5538596,
            "Ekurhuleni Metro": 3781377,
            "Sedibeng": 952102,
            "West Rand": 922640}

def get_run_data(predictions, data):
    config = predictions.replace('/run_', '/config_out_').replace('.csv', '.json')
    reader = open(config, 'r')
    config = json.load(reader)
    reader.close()
    predictions = pd.read_csv(predictions)

    pred_cases = []
    true_cases = []
    pred_hosp = []
    true_hosp = []
    for i, dist in enumerate(districts):
        n = districts[dist]
        mild = predictions[f'{dist}_Exposed'].values * n
        mild *= (1 - config['p1'][i]) / config['Incubation_Period'][i]
        pred_cases.append(mild)
        hosp = predictions[f'{dist}_Mild'].values * n
        hosp *= config['alpha_i2'][i]
        pred_hosp.append(hosp)

        true_cases.append(data[f'New Cases {dist}'].values)
        true_hosp.append(data[f'{dist} Hospitalised Cases'].values)

    pred_cases = sum(pred_cases)
    true_cases = sum(true_cases)
    pred_hosp = sum(pred_hosp)
    true_hosp = sum(true_hosp)

    pred = pred_cases + pred_hosp
    true = true_cases + true_hosp

    true_peak = true.argmax()
    pred_peak = pred.argmax()

    predictions = predictions.loc[(pred_peak - true_peak):].reset_index()
    predictions = predictions.loc[:len(data)]

    for i, dist in enumerate(districts):
        n = districts[dist]
        mild = predictions[f'{dist}_Exposed'] * n
        mild *= (1 - config['p1'][i]) / config['Incubation_Period'][i]
        data[f'{dist}_Mild'] = mild
        hosp = predictions[f'{dist}_Mild'] * n
        hosp *= config['alpha_i2'][i]
        data[f'{dist}_Hospitalised'] = hosp

    return data


if __name__ == '__main__':
    path = 'SENS_21'
    reader = open('best_ids.json', 'r')
    idx = json.load(reader)[::-1]
    reader.close()

    true_hosp = 'Data/wave1_pred_rep_hosp.csv'
    true_cases = 'Data/wave1_pred_rep_mild.csv'

    data_hosp = pd.read_csv(true_hosp)
    data_cases = pd.read_csv(true_cases)

    data = data_cases.join(data_hosp.set_index('Date'), on='Date')
    data = data.dropna().reset_index()

    predictions = f'{path}/run_{idx[0]}.csv'
    mean = get_run_data(predictions, deepcopy(data))

    all = [get_run_data(f'{path}/run_{i}.csv', deepcopy(data)) for i in idx[:100]]
    std = deepcopy(mean)
    cols = [f'{dist}_Mild' for dist in districts] + [f'{dist}_Hospitalised' for dist in districts]
    for col in cols:
        dat = [dat_[col].values.reshape(-1, 1) for dat_ in all]
        dat = np.concatenate(dat, 1)
        std[col] = dat.std(1) / np.sqrt(dat.shape[-1])
        #mean[col] = dat.mean(1)
    std = std[cols]
    
    width = 1.96
    fig, axs = plt.subplots(len(districts))
    for i, dist in enumerate(districts):
        pred = mean[f'{dist}_Mild'].values
        lower = pred - width * std[f'{dist}_Mild'].values
        lower[lower < 0.0] = 0.0
        upper = pred + width * std[f'{dist}_Mild'].values
        true = mean[f'New Cases {dist}'].rolling(window=7).mean().values

        axs[i].fill_between(mean['Date'].values, lower, upper, alpha=0.3)
        axs[i].plot(mean['Date'].values, pred)
        axs[i].plot(mean['Date'].values, true, linestyle='--')

    plt.show()

    fig, axs = plt.subplots(len(districts))
    for i, dist in enumerate(districts):
        pred = mean[f'{dist}_Hospitalised'].values
        lower = pred - width * std[f'{dist}_Hospitalised'].values
        lower[lower < 0.0] = 0.0
        upper = pred + width * std[f'{dist}_Hospitalised'].values
        true = mean[f'{dist} Hospitalised Cases'].rolling(window=7).mean().values

        axs[i].fill_between(mean['Date'].values, lower, upper, alpha=0.3)
        axs[i].plot(mean['Date'].values, pred)
        axs[i].plot(mean['Date'].values, true, linestyle='--')

    plt.show()