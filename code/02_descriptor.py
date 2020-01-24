#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file:
    linked with: .traj file @ ../tests/traj_folder/ and finger_f90 @ ../src/
    output: fingerprints file @ ../tests/fps_path/
'''

import os, math, argparse, hashlib, subprocess, sys
import numpy as np
import pickle, datetime
from ase import Atoms, Atom, units
from ase.io.trajectory import Trajectory
from amp.utilities import hash_images, get_hash
import shutil

def traj_info(atoms):
    """ Summary:
            function generates information of the atomistic system
            files written out @ given directory
    """
    f=open('cell_matrix', 'w+')
    f.write("%8.6f %8.6f %8.6f\n" % ( atoms.cell[0][0], atoms.cell[0][1], atoms.cell[0][2]))
    f.write("%8.6f %8.6f %8.6f\n" % ( atoms.cell[1][0], atoms.cell[1][1], atoms.cell[1][2]))
    f.write("%8.6f %8.6f %8.6f\n" % ( atoms.cell[2][0], atoms.cell[2][1], atoms.cell[2][2]))
    f.close()

    f=open('positions', 'w+')
    for atom in atoms.get_scaled_positions():
        f.write("%8.6f %8.6f %8.6f\n" % ( atom[0], atom[1], atom[2] ))
    f.close()

    f=open('elements', 'w+')
    for atom in atoms.get_chemical_symbols():
        f.write(atom + '\n')
    f.close()

    symbols=sorted(set(atoms.get_chemical_symbols()))
    f=open('element_alist', 'w+')
    for atom in symbols:
        f.write(atom + '\n')
    f.close()

def fp_generator(trajfile, Rc, calc_primes, fortran_path):
    """ Summary:
            generate fingerprints based on all traj files
        Parameters:
            trajfile: path of all.traj file
            Rc: cutoff radius
            calc_primes: 0: train energy
                         1: train both energy/force
            fortran_path: directory to compiled finger_f90
        Returns:
            writeout fingerprints files to the given folder
    """
    # generate descriptor path
    path_prime='amp-fingerprint-primes.ampdb/loose/'
    path_finger='amp-fingerprints.ampdb/loose/'
    path_neighbor='amp-neighborlists.ampdb/loose/'

    # check if paths exist
    if not os.path.exists(path_prime):
        os.makedirs(path_prime)
    if not os.path.exists(path_finger):
        os.makedirs(path_finger)
    if not os.path.exists(path_neighbor):
        os.makedirs(path_neighbor)

    # Reading all configurations
    traj = Trajectory(trajfile)
    for atoms in traj:
        hash = get_hash(atoms)
        print(hash)
        if (os.path.exists(path_neighbor + hash) and os.path.exists(path_finger + hash) and os.path.exists(path_prime + hash)):
            print(hash + "existed")
            continue
        else:
            print("Fingerprinting " + hash + " " + str(datetime.datetime.now()))
        # write out traj information
        traj_info(atoms) #$, 'cell_matrix', 'positions', 'elementlist', 'element_type')

        # call the fortran acceleration code
        symbols=sorted(set(atoms.get_chemical_symbols()))
        print(symbols)
        #command = "/Users/furinkazan/metal@zeo_codes/finger_f90" + " " + str(len(atoms)) + " " + str(len(symbols)) + " " + str(Rc) + " " + str(calc_primes)
        command = fortran_path + " " + str(len(atoms)) + " " + str(len(symbols)) + " " + str(Rc) + " " + str(calc_primes)
        print (command)

        #subprocess.call( "/Users/furinkazan/metal@zeo_codes/finger_f90" + " " + str(len(atoms)) + " " + str(len(symbols)) + " " + str(Rc) + " " + str(calc_primes), shell=True)
        subprocess.call(fortran_path + " " + str(len(atoms)) + " " + str(len(symbols)) + " " + str(Rc) + " " + str(calc_primes), shell=True)
        elem_list = []
        neighbors_list = []
        fingers = []
        if os.path.isfile('elements'):
            file=open('elements', 'r')
            for line in file:
                elem_list.append(line.strip())
                neighbors_list.append([[], []])
                fingers.append( (line.strip(), ) )
        else:
            print("elements file not exist")

        natoms=len(elem_list)
        ntypes=len(set(elem_list))

        # file neighbors and fingerprints generated from Fortran module
        if os.path.isfile('neighbors'):
            file = open('neighbors', 'r')
            for line in file:
                line_items=line.split()
                aa=int(line_items[0])-1
                bb=int(line_items[1])-1
                offset=[ int(line_items[2]), int(line_items[3]), int(line_items[4]) ]
                neighbors_list[aa][0].append(bb)
                neighbors_list[aa][1].append(offset)
            file.close()

        tup_list=[]
        for aa in range(natoms):
            tup1=( np.array(neighbors_list[aa][0]), np.array(neighbors_list[aa][1]) )
            tup_list.append(tup1)

        f = open(path_neighbor + hash, 'wb')
        pickle.dump(tup_list, f)
        f.close()

        if os.path.isfile('fingerprints'):
            jj=0
            kk=0
            nsymms=4*ntypes+4*(math.factorial(ntypes+1)/(2*math.factorial(ntypes-1)))
            ff=[]

            file = open('fingerprints', 'r')
            for line in file:
                ff.append(float(line))
                jj=jj+1
                if jj==nsymms:
                    fingers[kk]+=( ff, )
                    kk=kk+1
                    jj=0
                    ff=[]

        f = open(path_finger + hash, 'wb')
        pickle.dump(fingers, f)
        f.close()

        primes={}
        if os.path.isfile('primes'):
            file = open('primes', 'r')
            for line in file:
                line_items=line.split()
                mm=int(line_items[0])-1
                aa=int(line_items[1])-1
                ll=int(line_items[2])-1
                key=( aa, elem_list[aa], mm, elem_list[mm], ll )
                if key in primes:
                  primes[key].append(float(line_items[3]))
                else:
                    primes[key]=[float(line_items[3])]

        f = open(path_prime + hash, 'wb')
        pickle.dump(primes, f)
        f.close()

if __name__ == "__main__":
    trajfile = "../traj_folder/all.traj"
    fortran_path = "../src/finger_f90"
    fps_path = "../tests/fps_folder/" # folder to store fingerprints
    if os.path.exists(fps_path):
        shutil.rmtree(fps_path)
        os.makedirs(fps_path)
    else:
        os.makedirs(fps_path)
    os.chdir(fps_path)
    print("successfully redirect to @ {}".format(os.getcwd()))
    fp_generator(trajfile, 6.5, 1, fortran_path)
    print("All fingerprints generated")
