from pandas import read_csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as transforms

VARIABLES = ['Immunity_start', 'R0_lockdown_scale']

DIST_VARS = ['alpha_i23', 'gamma_i3', 'alpha_i24', 'delta_i3', 'gamma_i4',
            'delta_i4', 'gamma_i1', 'gamma_i2', 'R0', 'vulnerability', 'upsilon',
            'Exposed_start', 'mobility_level', 'Incubation_Period', 'Infective_Period', 'p1']

OUTCOMES = ['Mild', 'Hospitalised', 'Deaths']

DISTRICTS = ['City Of Tshwane Metro', 'City Of Johannesburg Metro', 'Ekurhuleni Metro', 'Sedibeng',
             'West Rand']

VAR_MAP = {'alpha_i23': 'Mild Hospitalisation Rate',
           'gamma_i3': 'Hospitalised Recovery Rate', 'alpha_i24': 'Mild ICU Rate',
           'delta_i3': 'Hospitalised Death Rate', 'gamma_i4': 'ICU Recovery Rate',
           'delta_i4': 'ICU Death Rate', 'gamma_i1': 'Asymptomatic Recovery Rate', 'gamma_i2': 'Mild Recovery Rate',
           'R0': 'Basic Reproductive Number', 'Immunity_start': 'Initial Immunity',
           'vulnerability': 'Vulnerability', 'vulnerability_normalised': 'Normalised Vulnerability',
           'upsilon': 'Daily Vaccination Rate', 'Adjusted_R0': 'Adjusted Basic Reproductive Number',
           'Exposed_start': 'Initial Exposed Population', 'mobility_level': 'Mobility Level',
           'R0_lockdown_scale': 'Lockdown Scaling', 'Incubation_Period': 'Incubation Period',
           'Infective_Period': 'Infective Period', 'p1': 'Percentage Asymptomatic Cases'}


def map_variables(vars, district):
    vars = [v.replace(f'{district}_', '') for v in vars]
    vars = [VAR_MAP.get(v, v) for v in vars]

    return vars


def get_district_statistics(data, district):
    vars = VARIABLES + [f'{district}_{v}' for v in DIST_VARS]
    outs = [f'{district}_{o}' for o in OUTCOMES]

    low_mild, high_mild = get_high_low(data, vars, outs[0])
    low_deaths, high_deaths = get_high_low(data, vars, outs[-1])

    impacts = np.abs(high_deaths - low_deaths)
    # impacts = high_deaths + low_deaths
    ordering = impacts.argsort()
    vars = np.array(vars)[ordering]
    low_deaths = low_deaths[ordering]
    high_deaths = high_deaths[ordering]
    low_mild = low_mild[ordering]
    high_mild = high_mild[ordering]

    avg_mild = data[outs[0]].values.mean()
    avg_deaths = data[outs[-1]].values.mean()

    return vars, low_mild, high_mild, low_deaths, high_deaths, avg_mild, avg_deaths


def get_high_low(data, variables, outcome):
    low, high = [], []
    outcome = data[outcome].values
    average = outcome.mean()
    for var in variables:
        vals = data[var].values
        # middle = (vals.max() - vals.min()) / 2 + vals.min()
        middle = np.median(vals)
        # middle = vals.mean()
        out_high = outcome[vals >= middle]
        out_low = outcome[vals < middle]

        low.append(out_low.mean())
        high.append(out_high.mean())
        # low.append(out_low.mean() - average)
        # high.append(average - out_high.mean())

    return np.array(low), np.array(high)


def tornado_plot(vars, low_mild, high_mild, low_deaths, high_deaths, avg_mild, avg_deaths):
    num_vars = len(vars)

    # bars centered on the y axis
    pos = np.arange(num_vars) + .5

    # make the left and right axes for women and men
    fig = plt.figure(facecolor='white', edgecolor='none')
    ax_low = fig.add_axes([0.05, 0.1, 0.35, 0.8])
    ax_high = fig.add_axes([0.6, 0.1, 0.35, 0.8])

    interval = 2000
    low_max = int(max(low_mild.max(), low_deaths.max()))
    high_max = int(max(high_mild.max(), high_deaths.max()))

    ax_low.set_xticks(np.arange(0, low_max + interval, interval))
    ax_low.set_xticklabels(np.arange(0, low_max + interval, interval), rotation=90)
    ax_high.set_xticks(np.arange(0, high_max + interval, interval))
    ax_high.set_xticklabels(np.arange(0, high_max + interval, interval), rotation=90)

    # turn off the axes spines except on the inside y-axis
    for loc, spine in ax_low.spines.items():
        if loc != 'right':
            spine.set_color('none') # don't draw spine

    for loc, spine in ax_high.spines.items():
        if loc != 'left':
            spine.set_color('none') # don't draw spine

    # just tick on the top
    ax_low.xaxis.set_ticks_position('top')
    ax_high.xaxis.set_ticks_position('top')

    # make the women's graphs
    ax_low.barh(pos, low_deaths, align='center',
                  facecolor='#DBE3C2', edgecolor='None')
    ax_low.barh(pos, low_mild, align='center', facecolor='#7E895F',
                  height=0.5, edgecolor='None')
    ax_low.axvline(avg_mild, color='black')
    ax_low.axvline(avg_deaths, color='black')
    ax_low.set_yticks([])
    ax_low.invert_xaxis()

    # make the men's graphs
    ax_high.barh(pos, high_deaths, align='center', facecolor='#D8E2E1',
                edgecolor='None')
    ax_high.barh(pos, high_mild, align='center', facecolor='#6D7D72',
                height=0.5, edgecolor='None')
    ax_high.axvline(avg_mild, color='black')
    ax_high.axvline(avg_deaths, color='black')
    ax_high.set_yticks([])

    # we want the cancer labels to be centered in the fig coord system and
    # centered w/ respect to the bars so we use a custom transform
    transform = transforms.blended_transform_factory(
        fig.transFigure, ax_high.transData)
    for i, label in enumerate(vars):
        ax_high.text(0.5, i+0.5, label, ha='center', va='center',
                    transform=transform)

    # the axes titles are in axes coords, so x=0, y=1.025 is on the left
    # side of the axes, just above, x=1.0, y=1.025 is the right side of the
    # axes, just above
    ax_high.set_title('HIGH', x=0.0, y=1.075, fontsize=12)
    ax_low.set_title('LOW', x=1.0, y=1.075, fontsize=12)

    # now add the annotations
    ax_high.annotate('Max Daily Mild Cases', xy=(0.95*high_mild[-1], num_vars-0.5),
                    xycoords='data',
                    xytext=(20, 0), textcoords='offset points',
                    size=12,
                    va='center',
                    arrowprops=dict(arrowstyle="->"),
                    )

    ax_high.annotate('Avg. Max Daily Mild Cases', xy=(avg_mild, 0),
                     xycoords='data',
                     xytext=(20, 0), textcoords='offset points',
                     size=12,
                     va='center',
                     arrowprops=dict(arrowstyle="->"),
                     )

    # a curved arrow for the deaths annotation
    ax_high.annotate('Total Deaths', xy=(0.7*high_deaths[-2], num_vars-1.5),
                    xycoords='data',
                    xytext=(20, 0), textcoords='offset points',
                    size=12,
                    va='center',
                    arrowprops=dict(arrowstyle="->"),
                    )

    ax_high.annotate('Avg. Total Deaths', xy=(avg_deaths, 1),
                     xycoords='data',
                     xytext=(20, 0), textcoords='offset points',
                     size=12,
                     va='center',
                     arrowprops=dict(arrowstyle="->"),
                     )

    return plt


if __name__ == '__main__':
    data = read_csv('SENS_50_results/sensitivity_data.csv')

    for district in DISTRICTS:
        vars, low_mild, high_mild, low_deaths, high_deaths, avg_mild, avg_deaths = get_district_statistics(data,
                                                                                                           district)
        vars = map_variables(vars.tolist(), district)
        plot = tornado_plot(vars, low_mild, high_mild, low_deaths, high_deaths, avg_mild, avg_deaths)
        plot.show()
