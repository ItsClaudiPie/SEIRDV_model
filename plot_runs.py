import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from copy import deepcopy
from matplotlib import pyplot as plt
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

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
        mild *= (1 - config['p1']) / config['Incubation_Period'][i]
        pred_cases.append(mild)
        hosp = predictions[f'{dist}_Mild'].values * n
        hosp *= config['alpha_i23'][i]
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

    predictions = predictions.loc[np.abs(pred_peak - true_peak):].reset_index()
    predictions = predictions.loc[:len(data)-1]

    for i, dist in enumerate(districts):
        n = districts[dist]
        mild = predictions[f'{dist}_Exposed'] * n
        mild *= (1 - config['p1']) / config['Incubation_Period'][i]
        data[f'{dist}_Mild'] = mild
        hosp = predictions[f'{dist}_Mild'] * n
        hosp *= config['alpha_i23'][i]
        data[f'{dist}_Hospitalised'] = hosp

    return data, config


if __name__ == '__main__':
    # parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--wave', type=int)
    # args = parser.parse_args()
    # wave = args.wave
    wave=1

    path = '/Users/carel17/Projects/SEIRDV_model/wave_1_0'

    true_hosp = f'/Users/carel17/Projects/SEIRDV_model/wave1_hosp.csv'
    true_cases = f'/Users/carel17/Projects/SEIRDV_model/wave1_mild.csv'

    data_hosp = pd.read_csv(true_hosp)
    data_cases = pd.read_csv(true_cases)

    data = data_cases.join(data_hosp.set_index('Date'), on='Date')
    data = data.dropna().reset_index()

    all = [get_run_data(f'{path}/run_{i + 1}.csv', deepcopy(data))[0] for i in range(5)]
    mean = deepcopy(all[0])
    lower = deepcopy(mean)
    upper = deepcopy(mean)
    cols = [f'{dist}_Mild' for dist in districts] + [f'{dist}_Hospitalised' for dist in districts]
    for col in cols:
        dat = [dat_[col].values.reshape(-1, 1) for dat_ in all]
        dat = np.concatenate(dat, 1)
        lower[col] = np.quantile(dat, 0.025, axis=1)
        upper[col] = np.quantile(dat, 0.975, axis=1)
        mean[col] = dat.mean(1)
    lower, upper = lower[cols], upper[cols]

    fig, axs = plt.subplots(len(districts))
    for i, dist in enumerate(districts):
        pred = mean[f'{dist}_Mild'].values
        low = lower[f'{dist}_Mild'].values
        upp = upper[f'{dist}_Mild'].values
        true = mean[f'New Cases {dist}'].rolling(window=7).mean().fillna(0.0).values

        axs[i].fill_between(mean['Date'].values, low, upp, alpha=0.3)
        axs[i].plot(mean['Date'].values, pred)
        axs[i].plot(mean['Date'].values, true, linestyle='--')
        axs[i].set_title(dist)
        axs[i].set_xticks([])
    n_ticks = 35
    plt.xticks(range(0, len(mean), n_ticks), mean['Date'][range(0, len(mean), n_ticks)])
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.show()
    # quit()

    # plt.savefig('/gpfs/project/niekerk/src/SEIRDV/SENS_24_results/mild.png')

    fig, axs = plt.subplots(len(districts))
    for i, dist in enumerate(districts):
        pred = mean[f'{dist}_Hospitalised'].values
        low = lower[f'{dist}_Hospitalised'].values
        upp = upper[f'{dist}_Hospitalised'].values
        true = mean[f'{dist} Hospitalised Cases'].rolling(window=7).mean().fillna(0.0).values

        axs[i].fill_between(mean['Date'].values, low, upp, alpha=0.3)
        axs[i].plot(mean['Date'].values, pred)
        axs[i].plot(mean['Date'].values, true, linestyle='--')

        axs[i].set_title(dist)
        axs[i].set_xticks([])
    plt.xticks(range(0, len(mean), n_ticks), mean['Date'][range(0, len(mean), n_ticks)])
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.show()

    # plt.savefig('/gpfs/project/niekerk/src/SEIRDV/SENS_24_results/hosp.png')