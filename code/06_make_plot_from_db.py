#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file:
    Making plot from stored database
"""

import sys
import shutil
import os
from ase.db import connect
sys.path.insert(0, '../src/')  # relative path
from plot_utils import fitting_energy_plot
from plot_utils import fitting_force_plot
from plot_utils import force_decompose_to_list

# change folder path
plot_path = "../01_demo_10nps/plots_from_db/"
if os.path.exists(plot_path):
    shutil.rmtree(plot_path)
    os.makedirs(plot_path)
else:
    os.makedirs(plot_path)
os.chdir(plot_path)

# read db
db_dir = "../../data/demo_02.db"
work_db = connect(db_dir)

db_tag_e_list = ['edt', 'edv', 'eat', 'eav']
db_e_dict = {}
for index, e_tag in enumerate(db_tag_e_list):
    e_list = []
    for row in work_db.select(tag=e_tag):
        atoms = work_db.get_atoms(id=row.id)
        e_atoms = atoms.get_potential_energy()
        e_list.append(e_atoms)
    db_e_dict[e_tag] = e_list

# print(db_e_dict)
db_tag_f_list = ['fdt', 'fdv', 'fat', 'fav']
db_f_dict = {}
for index, f_tag in enumerate(db_tag_f_list):
    f_list = []
    for row in work_db.select(tag=f_tag):
        atoms = work_db.get_atoms(id=row.id)
        f_atoms = atoms.get_forces()
        f_list.append(f_atoms)
    db_f_dict[f_tag] = f_list

for f_tag, f_atoms in db_f_dict.items():
    if 'dt' in f_tag:
        x_train_force_list, y_train_force_list, z_train_force_list = force_decompose_to_list(f_atoms)
    elif 'dv' in f_tag:
        x_valid_force_list, y_valid_force_list, z_valid_force_list = force_decompose_to_list(f_atoms)
    elif 'at' in f_tag:
        x_amp_train_force_list, y_amp_train_force_list, z_amp_train_force_list = force_decompose_to_list(f_atoms)
    elif 'av' in f_tag:
        x_amp_valid_force_list, y_amp_valid_force_list, z_amp_valid_force_list = force_decompose_to_list(f_atoms)
    else:
        raise Exception("Not key in db_f_dict found")

print(x_amp_train_force_list)
print(x_amp_valid_force_list)
"""
# energy plot
fitting_energy_plot(e_dft_train, e_dft_valid, e_amp_train, e_amp_valid, 'edft_v_eamp_wf')

# forces plot in x-y-z
fitting_force_plot(x_train_force_list, x_valid_force_list, x_amp_train_force_list, x_amp_valid_force_list,
'x_fdft_v_famp')
fitting_force_plot(y_train_force_list, y_valid_force_list, y_amp_train_force_list, y_amp_valid_force_list,
'y_fdft_v_famp')
fitting_force_plot(z_train_force_list, z_valid_force_list, z_amp_train_force_list, z_amp_valid_force_list,
'z_fdft_v_famp')

print("All plots generated, Task finished")
"""