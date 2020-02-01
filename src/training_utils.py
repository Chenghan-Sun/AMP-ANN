#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file:
    repo for utilities of Atomistic Machine Learning Project
    implemented classes:
        DatabaseTools: ab-initio data from vasprun.xml files
            --> db --> .trajfile
        FpsAnalysisTools:

        TrainTools:

"""

import sys
from ase import io
import numpy as np

from itertools import chain

# AMP Module
from amp import Amp
from amp.model.neuralnetwork import NeuralNetwork
from amp.regression import Regressor
from amp.model import LossFunction

from amp.analysis import plot_convergence

# Visual
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Sklearn module
sys.path.insert(0, "/usr/local/lib/python3.7/site-packages/")
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


class TrainTools:
    """ Summary:
            Training module using Machine learning for atomistic system model energies/forces fitting tasks
        Note:
            Use coupling with Fp_values_extractor method for generating descriptor.
    """
    def __init__(self, descriptor, path, force_option, training_traj='trainset.traj', validation_traj='validset.traj'):
        """ Parameters:
                Based on the fcn --> descriptor_generator in Fp_values_extractor module, returned descriptor
                    used in this fcn.
                force_option: choose if want to turn on force training
        """
        self.descriptor = descriptor
        self.path = path
        self.training_traj = training_traj
        self.validation_traj = validation_traj
        self.force_option = force_option

    def read_traj(self, trainset_index=':', validset_index=':'):
        """ Summary:
                read-in trajectories from traj_folder
            Parameters:
                .traj files input, specify selecting range of trajectories
            Returns:
                loaded train / test images
        """
        training_traj = io.read(self.path + self.training_traj, trainset_index)
        validation_traj = io.read(self.path + self.validation_traj, validset_index)
        return training_traj, validation_traj

    def train_amp_setup(self, trigger, training_traj, **kwargs):
        """ Summary:
                Adjusting convergence parameters
                how tightly the energy and/or forces are converged --> adjust the LossFunction
                To change how the code manages the regression process --> use the Regressor class
            Parameters:
                dictionary of neural-net parameters:
                dictionary tructure:
                    #1 'hiddenlayers': NN architecture
                    #2 'optimizer'
                    #3 'lossprime'：True if want gradient-based
                    #4 'convergence': convergence parameters
                    #5 force_coefficient: control the relative weighting of the energy and force RMSEs used
                        in the path to convergence
                    #6 indices_fit_forces: only use specified list of index of atoms for training,
                        worked with hacked model/__init__.py ver.
                trigger: control if choose to begin training
                training_traj: training set
            Returns:
                calc: Trained calculator

            TODO: #1 update optimizer
                  #2 update bootstrap-stat
                  #3 update more ML model
                  #4 update better method than decompose the nn_features_dict
        """
        nn_dict = [v for k, v in kwargs.items()]
        calc = Amp(descriptor=self.descriptor, model=NeuralNetwork(hiddenlayers=nn_dict[0], checkpoints=14),
                   label='amp')
        regressor = Regressor(optimizer=nn_dict[1], lossprime=nn_dict[2])
        calc.model.regressor = regressor
        if self.force_option is False:
            calc.model.lossfunction = LossFunction(convergence=nn_dict[3], )
        elif self.force_option is True:
            # calc.model.lossfunction = LossFunction(convergence=nn_dict[3], force_coefficient=nn_dict[4],
            #                                      indices_fit_forces=nn_dict[5])
            calc.model.lossfunction = LossFunction(convergence=nn_dict[3], force_coefficient=nn_dict[4])

        if trigger is True:
            calc.train(images=training_traj)
        else:
            print("Training NOT Start")

    @staticmethod
    def get_dft_energy(training_traj, validation_traj, rel_option):
        """ Inputs: train-valid sets from fcn --> read_traj
            Outputs: choose to return raw or/both relative dft energies
        """
        e_dft_train = np.array([atoms.get_potential_energy() for atoms in training_traj])
        e_dft_validation = np.array([atoms.get_potential_energy() for atoms in validation_traj])
        if rel_option == False:
            return e_dft_train, e_dft_validation
        elif rel_option == True:
            e_dft = np.concatenate((e_dft_train,e_dft_validation))
            rel_e_dft_train = e_dft_train - min(e_dft)
            rel_e_dft_validation = e_dft_validation - min(e_dft)
            return e_dft_train, e_dft_validation, rel_e_dft_train, rel_e_dft_validation

    def get_dft_force(self, training_traj, validation_traj, rel_option, norm_option, normalize_option):
        """ Inputs: train-valid sets from fcn --> read_traj
            Outputs: forces at X-Y-Z axises
                    choose to return raw or/both relative dft forces
                    choose to return total force
            TODO: using dictionary for linear search
        """
        if self.force_option is False:
            raise ValueError('Force_option is not turned on!')
        else:
            pass
        # assign DFT calculator
        f_dft_train = np.array([atoms.get_forces() for atoms in training_traj])
        f_dft_validation = np.array([atoms.get_forces() for atoms in validation_traj])

        # decompose forces into X-Y-Z axises
        x_train_force_list = []
        y_train_force_list = []
        z_train_force_list = []
        merged_f_dft_train = list(chain(*f_dft_train))
        for i in merged_f_dft_train:
            x_train_force_list.append(i[0])
            y_train_force_list.append(i[1])
            z_train_force_list.append(i[2])

        x_valid_force_list = []
        y_valid_force_list = []
        z_valid_force_list = []
        merged_f_dft_valid = list(chain(*f_dft_validation))
        for i in merged_f_dft_valid:
            x_valid_force_list.append(i[0])
            y_valid_force_list.append(i[1])
            z_valid_force_list.append(i[2])

        # normalize the decomposed forces
        if normalize_option is True:
            x_nor_train_force_list = preprocessing.normalize(x_train_force_list, norm='l2')
            y_nor_train_force_list = preprocessing.normalize(y_train_force_list, norm='l2')
            z_nor_train_force_list = preprocessing.normalize(z_train_force_list, norm='l2')
            x_nor_valid_force_list = preprocessing.normalize(x_valid_force_list, norm='l2')
            y_nor_valid_force_list = preprocessing.normalize(y_valid_force_list, norm='l2')
            z_nor_valid_force_list = preprocessing.normalize(z_valid_force_list, norm='l2')

        # translate to total force
        if norm_option is True:
            f_dft_train = [np.linalg.norm(forces) for forces in f_dft_train]
            f_dft_validation = [np.linalg.norm(forces) for forces in f_dft_validation]
            # option to get relative forces
            f_dft = np.concatenate((f_dft_train, f_dft_validation))
            rel_f_dft_tra = f_dft_train - min(f_dft)
            rel_f_dft_val = f_dft_validation - min(f_dft)
            return f_dft_train, f_dft_validation, rel_f_dft_tra, rel_f_dft_val

        elif norm_option is False and rel_option is True:
            f_x_dft = np.concatenate((x_train_force_list, x_valid_force_list))
            rel_f_x_dft_train = x_train_force_list - min(f_x_dft)
            rel_f_x_dft_valid = x_valid_force_list - min(f_x_dft)
            f_y_dft = np.concatenate((y_train_force_list, y_valid_force_list))
            rel_f_y_dft_train = y_train_force_list - min(f_y_dft)
            rel_f_y_dft_valid = y_valid_force_list - min(f_y_dft)
            f_z_dft = np.concatenate((z_train_force_list, z_valid_force_list))
            rel_f_z_dft_train = z_train_force_list - min(f_z_dft)
            rel_f_z_dft_valid = z_valid_force_list - min(f_z_dft)
            return (rel_f_x_dft_train, rel_f_x_dft_valid, rel_f_y_dft_train,
                    rel_f_y_dft_valid, rel_f_z_dft_train, rel_f_z_dft_valid)

        elif norm_option is False and rel_option is False and normalize_option is False:
            return (x_train_force_list, x_valid_force_list, y_train_force_list,
                    y_valid_force_list, z_train_force_list, z_valid_force_list)

        elif norm_option is False and rel_option is False and normalize_option is True:
            return (x_nor_train_force_list, x_nor_valid_force_list, y_nor_train_force_list,
                    y_nor_valid_force_list, z_nor_train_force_list, z_nor_valid_force_list)

    @staticmethod
    def get_neuralnet_energy(calc, training_traj, validation_traj, rel_option):
        """ Inputs: trained AMP calculator returned from fcn --> train_amp_setup
            Outputs: raw and/or relative AMP energies
        """
        e_amp_train = []
        e_amp_validation = []
        for atoms in training_traj:
            atoms.set_calculator(calc)
            e_train = atoms.get_potential_energy()
            e_amp_train.append(e_train)
        for atoms in validation_traj:
            atoms.set_calculator(calc)
            e_valid = atoms.get_potential_energy()
            e_amp_validation.append(e_valid)
        if rel_option is True:
            e_amp = np.concatenate((e_amp_train, e_amp_validation))
            rel_e_amp_train = e_amp_train - min(e_amp)
            rel_e_amp_validation = e_amp_validation - min(e_amp)
            return e_amp_train, e_amp_validation, rel_e_amp_train, rel_e_amp_validation
        elif rel_option is False:
            return e_amp_train, e_amp_validation

    def get_neuralnet_force(self, calc, training_traj, validation_traj, rel_option, norm_option, normalize_option):
        """ Inputs: trained AMP calculator
            Outputs: raw and relative AMP forces
            TODO: dictionary replacing list
        """
        if self.force_option is False:
            raise ValueError('Force_option is not turned on!')
        else:
            pass
        # assign AMP calculator
        f_amp_train = []
        f_amp_validation = []
        for atoms in training_traj:
            atoms.set_calculator(calc)
            f_train = atoms.get_forces()
            f_amp_train.append(f_train)
        for atoms in validation_traj:
            atoms.set_calculator(calc)
            f_valid = atoms.get_forces()
            f_amp_validation.append(f_valid)

        # decompose forces into X-Y-Z axises
        x_train_force_list = []
        y_train_force_list = []
        z_train_force_list = []
        merged_f_amp_train = list(chain(*f_amp_train))
        for i in merged_f_amp_train:
            x_train_force_list.append(i[0])
            y_train_force_list.append(i[1])
            z_train_force_list.append(i[2])

        x_valid_force_list = []
        y_valid_force_list = []
        z_valid_force_list = []
        merged_f_amp_valid = list(chain(*f_amp_validation))
        for i in merged_f_amp_valid:
            x_valid_force_list.append(i[0])
            y_valid_force_list.append(i[1])
            z_valid_force_list.append(i[2])

        # normalize the decomposed forces
        if normalize_option == True:
            x_nor_train_force_list = preprocessing.normalize(x_train_force_list, norm='l2')
            y_nor_train_force_list = preprocessing.normalize(y_train_force_list, norm='l2')
            z_nor_train_force_list = preprocessing.normalize(z_train_force_list, norm='l2')
            x_nor_valid_force_list = preprocessing.normalize(x_valid_force_list, norm='l2')
            y_nor_valid_force_list = preprocessing.normalize(y_valid_force_list, norm='l2')
            z_nor_valid_force_list = preprocessing.normalize(z_valid_force_list, norm='l2')

        # translate to total force
        if norm_option == True:
            f_amp_train = [np.linalg.norm(forces) for forces in f_amp_train]
            f_amp_validation = [np.linalg.norm(forces) for forces in f_amp_validation]
            # option to get relative forces
            f_amp = np.concatenate((f_amp_train, f_amp_validation))
            rel_f_amp_tra = f_amp_train - min(f_amp)
            rel_f_amp_val = f_amp_validation - min(f_amp)
            return f_amp_train, f_amp_validation, rel_f_amp_tra, rel_f_amp_val

        elif (norm_option == False and rel_option == True):
            f_x_amp = np.concatenate((x_train_force_list, x_valid_force_list))
            rel_f_x_amp_train = x_train_force_list - min(f_x_amp)
            rel_f_x_amp_valid = x_valid_force_list - min(f_x_amp)
            f_y_amp = np.concatenate((y_train_force_list, y_valid_force_list))
            rel_f_y_amp_train = y_train_force_list - min(f_y_amp)
            rel_f_y_amp_valid = y_valid_force_list - min(f_y_amp)
            f_z_amp = np.concatenate((z_train_force_list, z_valid_force_list))
            rel_f_z_amp_train = z_train_force_list - min(f_z_amp)
            rel_f_z_amp_valid = z_valid_force_list - min(f_z_amp)
            return (rel_f_x_amp_train, rel_f_x_amp_valid, rel_f_y_amp_train,
                    rel_f_y_amp_valid, rel_f_z_amp_train, rel_f_z_amp_valid)

        elif (norm_option == False and rel_option == False and normalize_option == False):
            return (x_train_force_list, x_valid_force_list, y_train_force_list,
                    y_valid_force_list, z_train_force_list, z_valid_force_list)

        elif (norm_option == False and rel_option == False and normalize_option == True):
            return (x_nor_train_force_list, x_nor_valid_force_list, y_nor_train_force_list,
                    y_nor_valid_force_list, z_nor_train_force_list, z_nor_valid_force_list)

    def fitting_energy_plot(self, e_dft_train, e_dft_validation, e_amp_train, e_amp_validation, fig_title):
        """ Inputs: train-valid dft/amp raw/rel energies
            Outputs: Figure 1: fitting results
                        blue circles: raw DFT dataset
                        black line: fitted values
                        orange circles: predicted values
                     Figure 2: Histogram of energy sampling
        """
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(e_dft_train, e_dft_train, '-k')
        plt.plot(e_dft_train, e_amp_train, 'o', fillstyle='full', color=colors[0],
                 label='training', markeredgecolor='k')
        plt.plot(e_dft_validation, e_amp_validation, 'o', fillstyle='full', color=colors[1],
                 label='validation', markeredgecolor='k')
        plt.legend()
        plt.xlabel('DFT energy, eV')
        plt.ylabel('ML-FF energy, eV')
        rms_valid = np.sqrt(mean_squared_error(e_dft_validation, e_amp_validation))
        rms_train = np.sqrt(mean_squared_error(e_dft_train, e_amp_train))
        #rms_valid_2 = np.sqrt(((e_amp_validation-e_dft_validation) ** 2).mean())
        #rms_train_2 = np.sqrt(((e_amp_train-e_dft_train) ** 2).mean())
        #print(rms_train_2, rms_valid_2)
        plt.title('RMSE train=%s, valid=%s' % (rms_train, rms_valid))

        plt.subplot(2, 1, 2)
        plt.hist(e_dft_train, color=colors[0], density=True, label='training')
        plt.hist(e_dft_validation, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
        plt.xlabel('DFT energy, eV')
        plt.ylabel('Occurence frequency')
        plt.legend()
        plt.savefig('./' + fig_title + '.png')
        return rms_train, rms_valid

    def fitting_force_plot(self, f_dft_train, f_dft_validation, f_amp_train, f_amp_validation, fig_title):
        """ Inputs: train-valid dft/amp raw/rel forces for a specific force axis
            Outputs: Figure 1: fitting results
                     Figure 2: Histogram of forces sampling
        """
        if self.force_option is False:
            raise ValueError('Force_option is not turned on!')
        else:
            pass

        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(f_dft_train, f_dft_train, '-k')
        plt.plot(f_dft_train, f_amp_train, 'o', fillstyle='full', color=colors[0],
                 label='training', markeredgecolor='k')
        plt.plot(f_dft_validation, f_amp_validation, 'o', fillstyle='full', color=colors[1],
                 label='validation', markeredgecolor='k')
        plt.legend()
        plt.xlabel('DFT force, eV/Ang')
        plt.ylabel('ML-FF force, eV/Ang')
        rms_valid = np.sqrt(mean_squared_error(f_dft_validation, f_amp_validation))
        rms_train = np.sqrt(mean_squared_error(f_dft_train, f_amp_train))
        plt.title('RMSE train=%s, valid=%s' % (rms_train, rms_valid))

        plt.subplot(2, 1, 2)
        plt.hist(f_dft_train, color=colors[0], density=True, label='training')
        plt.hist(f_dft_validation, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
        plt.xlabel('DFT force, eV/Ang')
        plt.ylabel('Occurence frequency')
        plt.legend()
        plt.savefig('./' + fig_title + '.png')
        return rms_train, rms_valid
