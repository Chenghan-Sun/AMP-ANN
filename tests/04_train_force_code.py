#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#SBATCH -J job -p skx-normal --nodes=1 -t 7:00:00 --output=job.out --error=job.err --ntasks-per-node=48

import time
start = time.time()
from ase import io
import matplotlib.pyplot as plt
import glob, os, sys
from ase.visualize import view
import shutil
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sys.path.insert(0,os.environ['HOME']+'08_hack_amp_model_test')
from training_utils import Fp_analysis_tools
from training_utils import Train_tools
from amp import Amp

'''
src = '/Users/furinkazan/00_testing_folder/backup_fps'
dst = '/Users/furinkazan/01_train_force'
src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)
print("successfully copied fp files")
'''

## main code ##
################################################################################
#Load Module
metal = 'Pt'
fp_analy = Fp_analysis_tools(metal, 1)
G2_etas = [0.05,4.0,20.0,80.0]
G4_etas = [0.005]
G4_zetas = [1., 4.]
G4_gammas = [+1., -1.]
descriptor, G = fp_analy.descriptor_generator(G2_etas, G4_etas, G4_zetas, G4_gammas)

path = "/Users/furinkazan/08_hack_amp_model_test/"
train_tools = Train_tools(descriptor, path, True) # turn force on
training_traj, validation_traj = train_tools.read_traj() #default index
e_dft_train, e_dft_validation = train_tools.get_dft_energy(training_traj, validation_traj, False)

x_train_force_list, x_valid_force_list, y_train_force_list, y_valid_force_list, z_train_force_list, z_valid_force_list = train_tools.get_dft_force(
training_traj, validation_traj, False, False, False)

nn_features_dict = {
    'hiddenlayers': (3,3,), # architecture
    'optimizer': 'L-BFGS-B',
    'lossprime': True,
    'convergence': {'energy_rmse': 0.02,
                    'force_rmse': 0.1},
    'force_coefficient': 0.04
}

print('*** START Training ***') # START Training

calc = train_tools.train_amp_setup(True, training_traj, **nn_features_dict)

print('*** END Training ***')

# if training ended
calc = Amp.load('amp.amp')

e_amp_train, e_amp_validation = train_tools.get_neuralnet_energy(calc, training_traj, validation_traj, False)
xamp_train_force_list, xamp_valid_force_list, yamp_train_force_list, yamp_valid_force_list, zamp_train_force_list, zamp_valid_force_list = train_tools.get_neuralnet_force(
calc, training_traj, validation_traj, False, False)

# Plot
train_tools.fitting_energy_plot(e_dft_train, e_dft_validation, e_amp_train, e_amp_validation, 'edft_v_eamp_wf')
train_tools.fitting_force_plot(x_train_force_list, x_valid_force_list, xamp_train_force_list, xamp_valid_force_list,
'x_fdft_famp')

train_tools.fitting_force_plot(y_train_force_list, y_valid_force_list, yamp_train_force_list, yamp_valid_force_list,
'y_fdft_famp')

train_tools.fitting_force_plot(z_train_force_list, z_valid_force_list, zamp_train_force_list, zamp_valid_force_list,
'z_fdft_famp')

end = time.time()
#print (end-start, file=open("running_time.txt", "a"))
f = open("running_time.txt", "a")
f.write("Code finished in " + str(end-start) + "s")
f.close()
