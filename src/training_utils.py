"""
This file:
    Repo for utilities of Atomistic Machine Learning Project
    implemented classes: TrainTools
    1. Learning Algorithm
    2. E/F calculations
    3. Plotting
    TODO: new version of plotting in plot_utils.py
"""

import sys
from ase import io
import numpy as np
from itertools import chain

# Database module
from ase.db import connect

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

    def get_energy_dict(self, calc, training_traj, validation_traj, db_dir, set_calc,
                        rel_option):
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
            e_train_dict: dictionary of DFT energies for training set
            e_valid_dict：dictionary of DFT energies for validation set
        """
        if not self.force_option:
            raise Exception("Force training is turned off for only energy training!")

        # check if old db exists
        new_db = connect(db_dir)  # assign new db's name
        for index, atoms in enumerate(training_traj):
            if not set_calc:  # if just DFT
                e_dft = atoms.get_potential_energy()  # assign DFT calculator
                self.e_train_dict[index] = e_dft  # update dictionary
                new_db.write(atoms, tag='edt')
            elif set_calc:  # training set & ML
                atoms.set_calculator(calc)  # assign ML calculator
                e_ml = atoms.get_potential_energy()
                self.e_train_dict[index] = e_ml
                new_db.write(atoms, tag='eat')
            else:
                raise Exception("Get Energies: training set calculator set-up is not specified!")

        for index, atoms in enumerate(validation_traj):
            if not set_calc:  # validation set & DFT
                e_dft = atoms.get_potential_energy()
                self.e_valid_dict[index] = e_dft
                new_db.write(atoms, tag='edv')
            elif set_calc:  # validation set & ML
                atoms.set_calculator(calc)
                e_ml = atoms.get_potential_energy()
                self.e_valid_dict[index] = e_ml
                new_db.write(atoms, tag='eav')
            else:
                raise Exception("Get Energies: validation set calculator set-up is not specified!")

        if not rel_option:
            return self.e_train_dict, self.e_valid_dict
        else:
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

    def get_forces_dict(self, calc, training_traj, validation_traj, db_dir, set_calc):
        """
        Summary:
            function to collect DFT / ML forces
        Params:
            calc: trained calculator by ML algorithms
            training_traj: training set
            validation_traj: validation set
            set_calc: flag for setting up trained calculator, default to be just DFT forces
        Returns:
            f_train_dict: dictionary of DFT forces for training set
            f_valid_dict：dictionary of DFT forces for validation set
        """
        if not self.force_option:  # check force training button opened
            raise ValueError('Force_option is not turned on for training both energy and force!')
        new_db = connect(db_dir)  # assign new db's name

        for index, atoms in enumerate(training_traj):
            if not set_calc:
                f_dft = atoms.get_forces()  # assign DFT calculator
                self.f_train_dict[index] = f_dft  # update dictionary
                new_db.write(atoms, tag='fdt')
            elif set_calc:
                atoms.set_calculator(calc)
                f_ml = atoms.get_forces()
                self.f_train_dict[index] = f_ml
                new_db.write(atoms, tag='fat')
            else:
                raise Exception("Get Forces: training set calculator set-up is not specified!")

        for index, atoms in enumerate(validation_traj):
            if not set_calc:
                f_dft = atoms.get_forces()  # assign DFT calculator
                self.f_valid_dict[index] = f_dft  # update dictionary
                new_db.write(atoms, tag='fdv')
            elif set_calc:
                atoms.set_calculator(calc)
                f_ml = atoms.get_forces()
                self.f_valid_dict[index] = f_ml
                new_db.write(atoms, tag='fav')
            else:
                raise Exception("Get Forces: validation set calculator set-up is not specified!")
        return self.f_train_dict, self.f_valid_dict

    def force_decompose(self, normalize):
        """
        Summary:
            Decompose forces into X-Y-Z axises
        TODO:
            rel_option: flag for relative forces
            norm_option: flag for total forces
            normalize: flag for normalized forces
        """
        x_train_force_list = []
        y_train_force_list = []
        z_train_force_list = []
        f_train_dict_val = [val for val in self.f_train_dict.values()]
        merged_f_train_dict_val = list(chain(*f_train_dict_val))
        for i in merged_f_train_dict_val:
            x_train_force_list.append(i[0])
            y_train_force_list.append(i[1])
            z_train_force_list.append(i[2])

        x_valid_force_list = []
        y_valid_force_list = []
        z_valid_force_list = []
        f_valid_dict_val = [val for val in self.f_valid_dict.values()]
        merged_f_valid_dict_val = list(chain(*f_valid_dict_val))
        for i in merged_f_valid_dict_val:
            x_valid_force_list.append(i[0])
            y_valid_force_list.append(i[1])
            z_valid_force_list.append(i[2])

        # normalize each decomposed forces
        if normalize:  # default normalize=False
            x_nor_train_force_list = preprocessing.normalize(np.array(x_train_force_list).reshape(1, -1), norm='l2')
            y_nor_train_force_list = preprocessing.normalize(np.array(y_train_force_list).reshape(1, -1), norm='l2')
            z_nor_train_force_list = preprocessing.normalize(np.array(z_train_force_list).reshape(1, -1), norm='l2')
            x_nor_valid_force_list = preprocessing.normalize(np.array(x_valid_force_list).reshape(1, -1), norm='l2')
            y_nor_valid_force_list = preprocessing.normalize(np.array(y_valid_force_list).reshape(1, -1), norm='l2')
            z_nor_valid_force_list = preprocessing.normalize(np.array(z_valid_force_list).reshape(1, -1), norm='l2')
            return x_nor_train_force_list.flatten(), x_nor_valid_force_list.flatten(), \
                y_nor_train_force_list.flatten(), y_nor_valid_force_list.flatten(), \
                z_nor_train_force_list.flatten(), z_nor_valid_force_list.flatten()

        else:
            return x_train_force_list, x_valid_force_list, y_train_force_list, y_valid_force_list, \
                   z_train_force_list, z_valid_force_list

    @staticmethod
    def fitting_energy_plot(e_dft_train, e_dft_valid, e_amp_train, e_amp_valid, fig_title):
        """
        Summary:
            train versus validation plot for energies
        Returns:
            Sub - Figure 1: fitting results
                blue circles: raw DFT dataset
                black line: fitted values
                orange circles: predicted values
            Sub - Figure 2: Histogram of energy sampling
        """
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(e_dft_train, e_dft_train, '-k')
        plt.plot(e_dft_train, e_amp_train, 'o', fillstyle='full', color=colors[0],
                 label='training', markeredgecolor='k')
        plt.plot(e_dft_valid, e_amp_valid, 'o', fillstyle='full', color=colors[1],
                 label='validation', markeredgecolor='k')
        plt.legend()
        plt.xlabel('DFT energies, eV')
        plt.ylabel('ML-FF energies, eV')
        rms_valid = np.sqrt(mean_squared_error(e_dft_valid, e_amp_valid))
        rms_train = np.sqrt(mean_squared_error(e_dft_train, e_amp_train))
        # rms_valid_2 = np.sqrt(((e_amp_validation-e_dft_validation) ** 2).mean())
        # rms_train_2 = np.sqrt(((e_amp_train-e_dft_train) ** 2).mean())
        # print(rms_train_2, rms_valid_2)
        plt.title('RMSE train=%s, valid=%s' % (np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)))

        plt.subplot(2, 1, 2)
        plt.hist(e_dft_train, color=colors[0], density=True, label='training')
        plt.hist(e_dft_valid, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
        plt.xlabel('DFT energies, eV')
        plt.ylabel('Occurrence frequency')
        plt.legend()
        plt.savefig('./' + fig_title + '.png')
        return np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)

    def fitting_force_plot(self, f_dft_train, f_dft_valid, f_amp_train, f_amp_valid, fig_title):
        """
        Summary:
            Plot the forces on x-y-z directions
        Returns:
            Sub - Figure 1: fitting results
                blue circles: raw DFT dataset
                black line: fitted values
                orange circles: predicted values
            Sub - Figure 2: Histogram of forces sampling
        """
        if not self.force_option:
            raise ValueError('Force_option is not turned on!')

        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(f_dft_train, f_dft_train, '-k')
        plt.plot(f_dft_train, f_amp_train, 'o', fillstyle='full', color=colors[0],
                 label='training', markeredgecolor='k')
        plt.plot(f_dft_valid, f_amp_valid, 'o', fillstyle='full', color=colors[1],
                 label='validation', markeredgecolor='k')
        plt.legend()
        plt.xlabel('DFT forces, eV/Ang')
        plt.ylabel('ML-FF forces, eV/Ang')
        rms_valid = np.sqrt(mean_squared_error(f_dft_valid, f_amp_valid))
        rms_train = np.sqrt(mean_squared_error(f_dft_train, f_amp_train))
        plt.title('RMSE train=%s, valid=%s' % (np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)))

        plt.subplot(2, 1, 2)
        plt.hist(f_dft_train, color=colors[0], density=True, label='training')
        plt.hist(f_dft_valid, color=colors[1], alpha=0.7, density=True, label='validation', ec='k')
        plt.xlabel('DFT forces, eV/Ang')
        plt.ylabel('Occurrence frequency')
        plt.legend()
        plt.savefig('./' + fig_title + '.png')
        return np.round(rms_train, decimals=3), np.round(rms_valid, decimals=3)
