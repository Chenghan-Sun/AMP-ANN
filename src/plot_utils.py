"""
This file:
    utils for making plot from stored database
"""

import sys
import numpy as np
from itertools import chain

# Visual
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# scikit-learn module
sys.path.insert(0, "/usr/local/lib/python3.7/site-packages/")
from sklearn.metrics import mean_squared_error


def fitting_energy_plot(e_dft_train, e_dft_valid, e_amp_train, e_amp_valid, fig_title):
    """
    Summary:
        train versus validation plot for energies
    Returns:
        Sub - Figure 1: fitting results
            blue circles: raw DFT dataset
            black line: fitted values
            orange circles: predicted values
        Sub - Figure 2: Histogram of energy sampling
    """
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.plot(e_dft_train, e_dft_train, '-k')
    plt.plot(e_dft_train, e_amp_train, 'o', fillstyle='full', color=colors[0],
             label='training', markeredgecolor='k')
    plt.plot(e_dft_valid, e_amp_valid, 'o', fillstyle='full', color=colors[1],
             label='validation', markeredgecolor='k')
    plt.legend()
    plt.xlabel('DFT energies, eV')
    plt.ylabel('ML-FF energies, eV')
    rms_valid = np.sqrt(mean_squared_error(e_dft_valid, e_amp_valid))
    rms_train = np.sqrt(mean_squared_error(e_dft_train, e_amp_train))
    # rms_valid_2 = np.sqrt(((e_amp_validation-e_dft_validation) ** 2).mean())
    # rms_train_2 = np.sqrt(((e_amp_train-e_dft_train) ** 2).mean())
    # print(rms_train_2, rms_valid_2)
    plt.title('RMSE train=%s, valid=%s' % (np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)))

    plt.subplot(2, 1, 2)
    plt.hist(e_dft_train, color=colors[0], density=True, label='training')
    plt.hist(e_dft_valid, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
    plt.xlabel('DFT energies, eV')
    plt.ylabel('Occurrence frequency')
    plt.legend()
    plt.savefig('./' + fig_title + '.png')
    return np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)


def force_decompose_to_list(f_dict_val):
    """
    Decompose forces in dictionary to lists of forces in xyz axises
    """
    x_force_list = []
    y_force_list = []
    z_force_list = []
    merged_f_dict_val = list(chain(*f_dict_val))
    for i in merged_f_dict_val:
        x_force_list.append(i[0])
        y_force_list.append(i[1])
        z_force_list.append(i[2])
    return x_force_list, y_force_list, z_force_list


def fitting_force_plot(f_dft_train, f_dft_valid, f_amp_train, f_amp_valid, fig_title):
    """
    Summary:
        Plot the forces on x-y-z directions
    Returns:
        Sub - Figure 1: fitting results
            blue circles: raw DFT dataset
            black line: fitted values
            orange circles: predicted values
        Sub - Figure 2: Histogram of forces sampling
    """
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.plot(f_dft_train, f_dft_train, '-k')
    plt.plot(f_dft_train, f_amp_train, 'o', fillstyle='full', color=colors[0],
             label='training', markeredgecolor='k')
    plt.plot(f_dft_valid, f_amp_valid, 'o', fillstyle='full', color=colors[1],
             label='validation', markeredgecolor='k')
    plt.legend()
    plt.xlabel('DFT forces, eV/Ang')
    plt.ylabel('ML-FF forces, eV/Ang')
    rms_valid = np.sqrt(mean_squared_error(f_dft_valid, f_amp_valid))
    rms_train = np.sqrt(mean_squared_error(f_dft_train, f_amp_train))
    plt.title('RMSE train=%s, valid=%s' % (np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)))

    plt.subplot(2, 1, 2)
    plt.hist(f_dft_train, color=colors[0], density=True, label='training')
    plt.hist(f_dft_valid, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
    plt.xlabel('DFT forces, eV/Ang')
    plt.ylabel('Occurrence frequency')
    plt.legend()
    plt.savefig('./' + fig_title + '.png')
    return np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)
