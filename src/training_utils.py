#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file:
    Repo for utilities of Atomistic Machine Learning Project
    implemented classes:
        `TrainTools:

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

# Visual
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# scikit-learn module
sys.path.insert(0, "/usr/local/lib/python3.7/site-packages/")
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


class TrainTools:
    """
    Summary:
        Training module using Machine learning for atomistic system model energies/forces fitting tasks
    Note:
        Use coupling with Fp_values_extractor method for generating descriptor.
    """
    def __init__(self, descriptor, path, force_option=True, training_traj='trainset.traj',
                 validation_traj='validset.traj'):
        """
        Summary:
            The params:
            force_option: bool. Choose if wants to turn on force training
        """
        self.descriptor = descriptor
        self.path = path
        self.training_traj = training_traj
        self.validation_traj = validation_traj
        self.force_option = force_option
        self.e_train_dict = {}
        self.e_valid_dict = {}
        self.f_train_dict = {}
        self.f_valid_dict = {}

    def read_traj(self, trainset_index=':', validset_index=':'):
        """
        Summary:
            read-in trajectories from traj_folder
        Params:
            .traj files input, specify selecting range of trajectories
        Returns:
            loaded train / test images
        """
        training_traj = io.read(self.path + self.training_traj, trainset_index)
        validation_traj = io.read(self.path + self.validation_traj, validset_index)
        return training_traj, validation_traj

    def train_amp_setup(self, trigger, training_traj, **kwargs):
        """
        Summary:
            Adjusting convergence parameters
            how tightly the energy and/or forces are converged --> adjust the LossFunction
            To change how the code manages the regression process --> use the Regressor class
        Params:
            dictionary of neural-net parameters:
            dictionary structure:
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
        nn_reg = Regressor(optimizer=nn_dict[1], lossprime=nn_dict[2])  # regressor
        calc.model.regressor = nn_reg
        if not self.force_option:
            calc.model.lossfunction = LossFunction(convergence=nn_dict[3], )
        elif self.force_option:
            calc.model.lossfunction = LossFunction(convergence=nn_dict[3], force_coefficient=nn_dict[4],
                                                   indices_fit_forces=nn_dict[5])

        if trigger is True:
            calc.train(images=training_traj)
        else:
            raise Exception("Training NOT Start")

    def get_energies(self, calc, training_traj, validation_traj, set_calc=False, rel_option=False):
        """
        Summary:
            function to collect DFT / ML energies
        Params:
            calc: trained calculator by ML algorithms
            training_traj: training set
            validation_traj: validation set
            set_calc: flag for setting up trained calculator, default to be just DFT energies
            rel_option: flag for get relative energies
        Returns:
            e_dft_train_dict: dictionary of DFT energies for training set
            e_dft_valid_dict：dictionary of DFT energies for validation set
        """
        if self.force_option:
            raise Exception("Force training is turned off for only energy training!")

        for index, atoms in enumerate(training_traj):
            if not set_calc:  # training set & DFT
                e_dft = atoms.get_potential_energy()  # assign DFT calculator
                self.e_train_dict[index] = e_dft  # update dictionary
            elif set_calc:  # training set & ML
                atoms.set_calculator(calc)  # assign ML calculator
                e_ml = atoms.get_potential_energy()
                self.e_train_dict[index] = e_ml
            else:
                raise Exception("Get Energies: training set calculator set-up is not specified!")

        for index, atoms in enumerate(validation_traj):
            if not set_calc:  # validation set & DFT
                e_dft = atoms.get_potential_energy()
                self.e_valid_dict[index] = e_dft
            elif set_calc:  # validation set & ML
                atoms.set_calculator(calc)
                e_ml = atoms.get_potential_energy()
                self.e_valid_dict[index] = e_ml
            else:
                raise Exception("Get Energies: validation set calculator set-up is not specified!")

        if rel_option:
            return self.e_train_dict, self.e_valid_dict
        elif not rel_option:
            min_e_dft_train = min(self.e_train_dict.items(), key=lambda x: x[1])[1]  # minimum value in dictionary
            min_e_dft_valid = min(self.e_valid_dict.items(), key=lambda x: x[1])[1]
            min_e_dft = min(min_e_dft_train, min_e_dft_valid)  # ground minimum
            rel_e_dft_train = [val - min_e_dft for val in self.e_train_dict.values()]
            rel_e_dft_valid = [val - min_e_dft for val in self.e_valid_dict.values()]

            # re update dictionary
            for key, val in enumerate(rel_e_dft_train):
                self.e_train_dict[key] = val
            for key, val in enumerate(rel_e_dft_valid):
                self.e_valid_dict[key] = val
            return self.e_train_dict, self.e_valid_dict

    def get_forces(self, calc, training_traj, validation_traj, set_calc=False, rel_option=False,
                   norm_option=False, normalize=False):
        """
        Summary:
            function to collect DFT / ML forces
        Params:
            calc: trained calculator by ML algorithms
            training_traj: training set
            validation_traj: validation set
            set_calc: flag for setting up trained calculator, default to be just DFT forces
            rel_option: flag for relative forces
            norm_option: flag for total forces
            normalize: flag for normalized forces
        Returns:

        """
        if not self.force_option:  # check force training button opened
            raise ValueError('Force_option is not turned on for training both energy and force!')

        for index, atoms in enumerate(training_traj):
            if not set_calc:
                f_dft = atoms.get_forces()  # assign DFT calculator
                self.f_train_dict[index] = f_dft  # update dictionary
            elif set_calc:
                atoms.set_calculator(calc)
                f_ml = atoms.get_forces()
                self.f_train_dict[index] = f_ml
            else:
                raise Exception("Get Forces: training set calculator set-up is not specified!")

        for index, atoms in enumerate(validation_traj):
            if not set_calc:
                f_dft = atoms.get_forces()  # assign DFT calculator
                self.f_valid_dict[index] = f_dft  # update dictionary
            elif set_calc:
                atoms.set_calculator(calc)
                f_ml = atoms.get_forces()
                self.f_valid_dict[index] = f_ml
            else:
                raise Exception("Get Forces: validation set calculator set-up is not specified!")

        # decompose forces into X-Y-Z axises
        x_train_force_list = []
        y_train_force_list = []
        z_train_force_list = []
        merged_f_train_dict_val = list(chain(*self.f_train_dict.values()))
        for i in merged_f_train_dict_val:
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
        if normalize is True:
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

        elif norm_option is False and rel_option is False and normalize is False:
            return (x_train_force_list, x_valid_force_list, y_train_force_list,
                    y_valid_force_list, z_train_force_list, z_valid_force_list)

        elif norm_option is False and rel_option is False and normalize is True:
            return (x_nor_train_force_list, x_nor_valid_force_list, y_nor_train_force_list,
                    y_nor_valid_force_list, z_nor_train_force_list, z_nor_valid_force_list)


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
        # rms_valid_2 = np.sqrt(((e_amp_validation-e_dft_validation) ** 2).mean())
        # rms_train_2 = np.sqrt(((e_amp_train-e_dft_train) ** 2).mean())
        # print(rms_train_2, rms_valid_2)
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
        """
        Summary:
            Plot the forces on x-y-z directions
        Parameters:
            train-valid dft/amp raw/rel forces for a specific force axis
        Returns:
            Figure 1: fitting results
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
