#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from ase import Atoms, Atom, units
import ase.io
from ase.build import fcc110

# Visual
from ase.visualize import view
from matplotlib import pyplot as plt

import shutil
sys.path.insert(0,"/Users/furinkazan/01_train_force")
from training_utils import Fp_analysis_tools

## Main Code ##
################################################################################
# remove the lastest amp* folders
try:
    shutil.rmtree("amp-data-fingerprints.ampdb", "amp-data-neighborlists.ampdb")
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

#read-in .traj file
global images
training_traj = ase.io.read('./trainset.traj',':')
images = [training_traj[1],training_traj[2]]

metal = 'Pt'
fp_analy = Fp_analysis_tools(metal, 1)
G2_etas = [0.05,4.0,20.0, 80.0]
G4_etas = [0.005]
G4_zetas = [1., 4.]
G4_gammas = [+1., -1.]
descriptor, G = fp_analy.descriptor_generator(G2_etas, G4_etas, G4_zetas, G4_gammas)

#start fingerprints Analysis
descriptor, images = fp_analy.calc_fp(descriptor, images)

#print out information about the fingerprint / the scatter plot of the fp values
for index, hash in enumerate(images.keys()):
    #fp_analy.fp_values_extractor(0, G, hash, descriptor, "scatter plot of the fp values.png")
    fp_analy.fp_barplot(0, G, hash, descriptor, 'bplot-%02i.png' % index,
    "bar plot of the fp values")

#test extreme values
max_fp, min_fp = fp_analy.get_extreme_value_fp(descriptor, images)

""" Some notes about the fp data structure:
    The key of fp --> each of the atoms, 43 in total
        the hash of fp: 0 --> the element type of atom e.g. oxygen
        1 --> each fp values, 36 in total, refering to the Gaussian data
    if fp = descriptor.fingerprints[hash]:
        fp gives # of images, each image contains list of all 43 elements, each element
        has 36 fp values, which could be tested by:
        fp = descriptor.fingerprints[hash][atom_index][1]
        print(fp, len(fp)))
    The number of fps caculation rule:
        G2: 4 eta-parameters X 3 different type of atoms = 12 fps
        G4: 1 eta-parameter X 2 zeta-parameter X 2 gamma-parameter X 6 (A3_2) types of
        2-atom interactions = 24 fps
        --> all together 36 fps
"""
