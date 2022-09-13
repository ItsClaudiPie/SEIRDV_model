import json
import pandas as pd

params = ['vulnerability', 'R0', 'R0_lockdown_scale', 'upsilon', 'p1', 'alpha_i2',
          'gamma_i3', 'alpha_i3', 'delta_i3', 'gamma_i4', 'delta_i4', 'mobility_level',
          'Incubation_Period', 'Infective_Period', 'Exposed_start', 'Immunity_start',
          'gamma_i1', 'gamma_i2']


def get_params(idx):
    config = f'/gpfs/project/niekerk/src/SEIRDV/SENS_23/config_out_{idx}.json'
    reader = open(config, 'r')
    config = json.load(reader)
    reader.close()

    parameters = []
    columns = []
    for param in params:
        vals = config[param]
        if type(vals) != list:
            vals = [vals] * len(config['districts'])
        parameters += vals
        columns += [f'{param}_{dist}' for dist in config['districts']]

    return pd.DataFrame([parameters], columns=columns)


def get_wave(wave):
    best_ids = f'/gpfs/project/niekerk/src/SEIRDV/SENS_24_results/best_ids_wave{wave}.json'
    reader = open(best_ids, 'r')
    best_ids = json.load(reader)
    reader.close()

    best_params = [get_params(i) for i in best_ids[:25]]
    best_params = pd.concat(best_params, 0).reset_index(drop=True)

    std = best_params.std(0)
    mean = best_params.loc[0]
    lower = mean.values - 1.96 * std.values
    upper = mean.values + 1.96 * std.values
    best_params = pd.DataFrame([mean.values, std.values, lower, upper], columns=mean.keys())
    best_params['Stat'] = ['mean', 'std', 'lower', 'upper']
    best_params['Wave'] = wave

    return best_params[['Wave', 'Stat'] + [key for key in mean.keys()]]


if __name__ == '__main__':
    waves = [get_wave(i) for i in [1, 2, 3, 4, 5]]
    waves = pd.concat(waves, 0).reset_index(drop=True)

    waves.to_csv('/gpfs/project/niekerk/src/SEIRDV/SENS_24_results/parameters.csv', index=False)
