import json
import os

import pandas as pd
from tqdm import tqdm


def load_data(path, seed):
    reader = open(os.path.join(path, f'config_out_{seed}.json'), 'r')
    config = json.load(reader)
    reader.close()

    data = pd.read_csv(os.path.join(path, f'run_{seed}.csv'))
    return config, data


def extract_variables(config):
    districts = config['districts']

    variables = ['Immunity_start', 'R0_lockdown_scale']
    values = [config[var] for var in variables]
    dist_vars = ["alpha_i23", "gamma_i3", "alpha_i24", "delta_i3", "gamma_i4", "delta_i4",
                 "gamma_i1", "gamma_i2", 'R0', 'vulnerability', 'vulnerability_normalised',
                 'upsilon', 'Adjusted_R0', 'Exposed_start', 'mobility_level',
                 'Incubation_Period', 'Infective_Period', 'p1']
    for variable in dist_vars:
        variables += [f'{dist}_{variable}' for dist in districts]
        values += [config[variable][i] for i, dist in enumerate(districts)]

    return pd.DataFrame([values], columns=variables), districts, config['N']


def extract_stats(data, districts, population_sizes):
    variables = [f'{dist}_Mild' for dist in districts]
    variables += [f'{dist}_Hospitalised' for dist in districts]
    variables += [f'{dist}_Deaths' for dist in districts]

    stats = []
    for var in variables:
        stat = data[var].max()
        stats.append(stat * population_sizes[districts.index(var.split('_', 1)[0])])

    return pd.DataFrame([stats], columns=variables)


def combine_seeds(path):
    runs = [file for file in os.listdir(path) if 'run_' in file]
    seeds = [int(run.split('_', 1)[1].replace('.csv', '')) for run in runs]

    runs2 = [file for file in os.listdir(path) if 'config_out_' in file]
    seeds2 = [int(run.split('_', 2)[-1].replace('.json', '')) for run in runs2]
    seeds = [seed for seed in seeds if seed in seeds2]

    stats = []
    for seed in tqdm(seeds):
        config, data = load_data(path, seed)
        variables, districts, population_sizes = extract_variables(config)
        stats_ = extract_stats(data, districts, population_sizes)
        stats_ = pd.concat(objs=(variables, stats_), axis=1)
        stats.append(stats_)

    return pd.concat(objs=stats, axis=0)


if __name__ == "__main__":
    path = '/gpfs/project/niekerk/src/SEIRDV/SENS_50'
    data = combine_seeds(path)
    path = f'{path}_results'
    data.to_csv(os.path.join(path, 'sensitivity_data.csv'), index=False)
