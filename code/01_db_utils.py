#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file:
    returns all, training-set, test-set .traj files from selected database
    note: use relative paths along this repo
    Inputs for using the script:
        db_name: choose which db to use
        db_features_dict: tag of selected db
        traj: subset of db selection
        training_weight: in db_tools.train_test_split()
        test_folder: name of demo folder
    Output: traj files written @ "../tests/traj_folder/"
"""

import os
import sys
from ase import io
sys.path.insert(0, '../src/')  # relative path
from training_utils import DatabaseTools

db_path = '../data/'
# db_name = 'md_sod_pt.db'
db_name = '00_100_spe_7pt.db'

# @linux: ase db 00_100_spe_7pt.db "fmax<2.5" "energy<-27" "zeo=spe_7pt"
db_features_dict = {
    "zeo": 'zeo',  # fed-in the key itself
    # "fmax": 'fmax<3.0',
    # "energy": 'energy<-26.5',
    # "small_dataset": 'small_dataset=True'
}

# load class
db_tools = DatabaseTools(db_path, db_name)
traj = db_tools.db_selector(**db_features_dict)[0:15]  # create a small traj file as demo
len_traj, traj_train, traj_valid = db_tools.train_test_split(traj, 0.67)

print('Images_total=', len_traj)
print('Images_validation=', len(traj_train))
print('Images_training=', len(traj_valid))

test_folder = "01_demo_10nps/"
traj_path = "../" + test_folder + "traj_folder/"
os.makedirs(traj_path)
os.chdir(traj_path)

if os.path.isfile('all.traj'):
    os.remove('all.traj')
    os.remove('trainset.traj')
    os.remove('validset.traj')
    print("deleted old traj files")
else:
    print('No existing traj files detected')

io.write('./all.traj', traj)
io.write('./trainset.traj', traj_train)
io.write('./validset.traj', traj_valid)

print("successfully write traj files @ {}".format(os.getcwd()))
