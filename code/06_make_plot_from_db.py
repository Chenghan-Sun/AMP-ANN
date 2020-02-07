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

print(db_e_dict)

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