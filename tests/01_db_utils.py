#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file:
    returns all, training-set, test-set .traj files from selected database
    note: use relative paths along the repo
'''

import glob, os, sys
from ase import io
import numpy as np
import shutil

sys.path.insert(0, '../src/') # relative path
from training_utils import Database_tools

# @linux: ase db 00_100_spe_7pt.db "fmax<2.5" "energy<-27" "zeo=spe_7pt"
db_features_dict = {
    "zeo": 'zeo', # fed-in the key itself
    "fmax": 'fmax<3.0',
    "energy": 'energy<-26.5',
    "small_dataset": 'small_dataset=True'
}
db_path = '../data/'
db_name = 'md_sod_pt.db'

# Load Module
db_tools = Database_tools(db_path, db_name)
traj = db_tools.db_selector(**db_features_dict)[0:15] # create a small traj file as demo
len_traj, traj_train, traj_valid = db_tools.train_test_split(True, traj, 0.67)

print('Images_total=', len_traj)
print('Images_validation=', len(traj_train))
print('Images_training=', len(traj_valid))

traj_path = "../traj_folder/"
os.makedirs(traj_path)
os.chdir(traj_path)

if os.path.isfile('all.traj'):
    os.remove('all.traj')
    os.remove('trainset.traj')
    os.remove('validset.traj')
    print("delete old traj files")
else:
    print('No existing traj files detected')

io.write('./all.traj', traj)
io.write('./trainset.traj', traj_train)
io.write('./validset.traj', traj_valid)

print("successfully write traj files @ {}".format(os.getcwd()))
