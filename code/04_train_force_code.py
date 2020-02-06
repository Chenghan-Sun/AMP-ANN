#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This file:
        training script for train on both energy/force tasks
        Inputs:
            test_folder: name of demo folder
            fp_analy: fingerprint analysis module, specify what type of metal, w/wo Zeolite framework
            G2_etas, G4_etas, G4_zetas, G4_gammas: Gaussian parameters
            nn_features_dict:
                NN architecture
                energy / force cutoff
                indices_fit_forces
        output:
            calculator amp.amp @ "../XXX/fps_folder"
            where XXX = demo folder
"""

import time
import os
import sys
import shutil
import numpy as np
from colorama import Fore, Style

sys.path.insert(0, '../src/')  # relative path
from fps_utils import FpsAnalysisTools
from training_utils import TrainTools
from amp import Amp

test_folder = "01_demo_10nps/"  # make the job testing folder
fps_path = "../" + test_folder + "fps_folder/"
os.chdir(fps_path)
print("Redirect to fps folder @ " + str(os.getcwd()))

# load class
fp_analy = FpsAnalysisTools('Pt', 0)  # only the Pt nano-cluster
G2_etas = [0.05, 4.0, 20.0, 80.0]
G4_etas = [0.005]
G4_zetas = [1., 4.]
G4_gammas = [+1., -1.]
descriptor, G = fp_analy.descriptor_generator(G2_etas, G4_etas, G4_zetas, G4_gammas)

path = "../traj_folder/"  # linking traj_folder from the fps_folder
train_tools = TrainTools(descriptor, path, True)  # turn force on
training_traj, validation_traj = train_tools.read_traj()  # use default index

""" get dft energies 
Note: 
    None: no calc needed, use with first bool=F
    first bool: T: set AMP calc; F: DFT calc
    second bool: T: relative data; F: raw data
"""
db_dir = "../../data/demo_02.db"
e_dft_train_dict, e_dft_valid_dict = train_tools.get_energy_dict(None, training_traj, validation_traj,
                                                                 db_dir, False, False)
e_dft_train = np.array([val for val in e_dft_train_dict.values()])
e_dft_valid = np.array([val for val in e_dft_valid_dict.values()])

""" get dft forces
Note: 
    None: no calc needed, use with first bool=F
    first bool: F: DFT calc
    second bool: T: relative data; F: raw data
    get amp energies / forces
Note:
    calc = calc_trained for loading trained calculator from ML
    first bool: T: set AMP calc
"""
f_dft_train_dict, f_dft_valid_dict = train_tools.get_forces_dict(None, training_traj, validation_traj, db_dir, False)

x_train_force_list, x_valid_force_list, y_train_force_list, y_valid_force_list, z_train_force_list, \
    z_valid_force_list = train_tools.force_decompose(False)

# Neural Network features
fitting_list = 'all'  # if specify training atoms subset using hacked model/__init__.py
nn_features_dict = {
    'hiddenlayers': (3, 3, ),  # architecture
    'optimizer': 'L-BFGS-B',
    'lossprime': True,
    'convergence': {'energy_rmse': 0.02,  # e-cutoff
                    'force_rmse': 0.16},  # f-cutoff
    'force_coefficient': 0.04,
    'indices_fit_forces': fitting_list  # '[0, 1, 2]
}

# training section
print(f"*** {Fore.GREEN}START Training{Style.RESET_ALL} ***")
start = time.time()
train_tools.train_amp_setup(True, training_traj, **nn_features_dict)
end = time.time()
f = open("../running_time.txt", "a")
f.write("Code finished in " + str(end-start) + "s")
f.close()
print(f"*** {Fore.GREEN}END Training{Style.RESET_ALL} ***")

# load calculator
calc_trained = Amp.load('amp.amp')
e_amp_train_dict, e_amp_valid_dict = train_tools.get_energy_dict(calc_trained, training_traj,
                                                                 validation_traj, db_dir, True, False)
e_amp_train = np.array([val for val in e_amp_train_dict.values()])
e_amp_valid = np.array([val for val in e_amp_valid_dict.values()])

f_amp_train_dict, f_amp_valid_dict = train_tools.get_forces_dict(calc_trained, training_traj,
                                                                 validation_traj, db_dir, True)

x_amp_train_force_list, x_amp_valid_force_list, y_amp_train_force_list, y_amp_valid_force_list, \
    z_amp_train_force_list, z_amp_valid_force_list = train_tools.force_decompose(False)

print(f"{Fore.RED}Task finished{Style.RESET_ALL}")

""" If still want to plot here 
plot_path = "../plots/"
if os.path.exists(plot_path):
    shutil.rmtree(plot_path)
    os.makedirs(plot_path)
else:
    os.makedirs(plot_path)
os.chdir(plot_path)

train_tools.fitting_energy_plot(e_dft_train, e_dft_valid, e_amp_train, e_amp_valid, 'edft_v_eamp_wf')

train_tools.fitting_force_plot(x_train_force_list, x_valid_force_list, x_amp_train_force_list, x_amp_valid_force_list,
'x_fdft_v_famp')
train_tools.fitting_force_plot(y_train_force_list, y_valid_force_list, y_amp_train_force_list, y_amp_valid_force_list,
'y_fdft_v_famp')
train_tools.fitting_force_plot(z_train_force_list, z_valid_force_list, z_amp_train_force_list, z_amp_valid_force_list,
'z_fdft_v_famp')

print("All plots generated, Task finished")
"""
