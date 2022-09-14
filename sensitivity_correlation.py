import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

data = pd.read_csv('SENS_50_results/sensitivity_data.csv')

districts = [dist.split('_', 1)[0] for dist in data.keys() if 'Deaths' in dist]

VAR_MAP = {'alpha_i1': 'Asymptomatic to Mild Rate', 'alpha_i23': 'Mild Hospitalisation Rate',
           'gamma_i3': 'Hospitalised Recovery Rate', 'alpha_i24': 'Hospitalised ICU Rate',
           'delta_i3': 'Hospitalised Death Rate', 'gamma_i4': 'ICU Recovery Rate',
           'delta_i4': 'ICU Death Rate', 'gamma_i1': 'Asymptomatic Recovery Rate', 'gamma_i2': 'Mild Recovery Rate',
           'R0': 'Basic Reproductive Number', 'Immunity_start': 'Initial Immunity',
           'vulnerability': 'Vulnerability', 'vulnerability_normalised': 'Normalised Vulnerability',
           'upsilon': 'Daily Vaccination Rate', 'Adjusted_R0': 'Adjusted Basic Reproductive Number',
           'Exposed_start': 'Initial Exposed Population', 'mobility_level': 'Mobility Level',
           'R0_lockdown_scale': 'Lockdown Scaling', 'Incubation_Period': 'Incubation Period',
           'Infective_Period': 'Infective Period', 'p1': 'Percentage Asymptomatic Cases'}

for dist in districts:
    variables = ['Immunity_start', 'R0_lockdown_scale']
    dist_vars = ['alpha_i23', 'gamma_i3', 'alpha_i24', 'delta_i3', 'gamma_i4',
                'delta_i4', 'gamma_i1', 'gamma_i2', 'R0', 'vulnerability', 'upsilon',
                'Exposed_start', 'mobility_level', 'Incubation_Period', 'Infective_Period', 'p1']

    variables += [f'{dist}_{var}' for var in dist_vars]
    outcomes = [f'{dist}_{o}' for o in ['Mild', 'Hospitalised', 'Deaths']]

    corr = np.corrcoef(data[variables + outcomes].values.T)
    corr = corr[:len(variables), len(variables):]

    impact = np.abs(corr).sum(-1)
    impact = impact.argsort()[::-1]
    corr = corr[impact].T

    plt.figure()
    plt.matshow(corr, vmin=-np.abs(corr.max().max()), vmax=np.abs(corr.max().max()), cmap='seismic')

    for (i, j), z in np.ndenumerate(corr):
        color = 'black' if np.abs(z) < 0.2 else 'white'
        plt.text(j, i, f'{round(z, 3)}', ha='center', va='center', color=color)

    outcomes = [o.replace(f'{dist}_', '') for o in outcomes]
    variables = [v.replace(f'{dist}_', '') for v in variables]
    variables = [VAR_MAP.get(v, v) for v in variables]
    variables = np.array(variables)[impact]

    plt.yticks(range(len(outcomes)), outcomes)
    plt.xticks(range(len(variables)), variables, rotation=90)
    plt.colorbar()
    plt.savefig(f'{dist}_sens.png', dpi=300, bbox_inches = "tight")
    plt.close()
