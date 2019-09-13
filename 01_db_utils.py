#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os, sys
from ase import io
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,"/Users/furinkazan/01_train_force")
from training_utils import Database_tools

## main code ##
################################################################################
# linux --> ase db 00_100_spe_7pt.db "fmax<2.5" "energy<-27" "zeo=spe_7pt"

db_features_dict = {
    "zeo": 'zeo=spe_7pt',
    "fmax": 'fmax<2.5',
    "energy": 'energy<-26.0'
}

db_path = '/Users/furinkazan/Box/work/QE_data/01_final_dbs/02_spe_7pt_nps/'
db_name = '00_100_spe_7pt.db'

db_tools = Database_tools(db_path, db_name)
traj = db_tools.db_selector(**db_features_dict)

len_traj, traj_train, traj_valid = db_tools.train_test_split(True, traj, 0.66)


print('Images_total=', len_traj)
print('Images_validation=', len(traj_train))
print('Images_training=', len(traj_valid))

io.write('./all.traj', traj)
io.write('./trainset.traj', traj_train)
io.write('./validset.traj', traj_valid)
