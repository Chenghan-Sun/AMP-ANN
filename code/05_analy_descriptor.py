#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file:
        Analyze fingerprints
"""
import os, sys
import numpy as np
import shutil
import ase

# Visual
from colorama import Fore, Style

sys.path.insert(0, '../src/')  # relative path
from training_utils import Fp_analysis_tools

''' remove the lastest amp* folders
try:
    shutil.rmtree("amp-data-fingerprints.ampdb", "amp-data-neighborlists.ampdb")
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))
'''

traj_path = "../tests/traj_folder/"
os.chdir(traj_path)  # redirect to fps folder

# read-in .traj file
global images
training_traj = ase.io.read('./trainset.traj',':')
images = [training_traj[1],training_traj[2]]

fp_analy = Fp_analysis_tools('Pt', 0)  # only the Pt nanocluster
G2_etas = [0.05,4.0,20.0, 80.0]
G4_etas = [0.005]
G4_zetas = [1., 4.]
G4_gammas = [+1., -1.]
descriptor, G = fp_analy.descriptor_generator(G2_etas, G4_etas, G4_zetas, G4_gammas)

# start fingerprints Analysis
descriptor, hashimages = fp_analy.calc_fp(descriptor, images)

# print out information about the fingerprint / the scatter plot of the fp values
for index, hash in enumerate(hashimages.keys()): # for all images
    print("hash = {}:".format(index), "\033[92m" + hash + "\033[0m")
    fp_analy.fp_values_extractor(0, G, hash, descriptor,)
    # fp_analy.fp_barplot(0, G, hash, descriptor, 'bplot-%02i.png' % index,
    # "bar plot of the fp values")
