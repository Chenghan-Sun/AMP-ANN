# Database module
from ase.db import connect

# Math module
from random import shuffle
import numpy as np

# ASE module
from ase import io
import os


class DatabaseTools:
    """
    Summary:
        ASE Database tools used for sampling ab-initio data
    init:
        specify directory path of database, database name
        default all training-validation .traj file names
    """
    def __init__(self, db_path, db_name, dataset='all.traj',
                 training_traj='trainset.traj', validation_traj='validset.traj'):
        self.path = db_path
        self.name = db_name
        self.dataset = dataset
        self.training_traj = training_traj
        self.validation_traj = validation_traj

    def xml_to_db(self, fmax_cutoff, zeo_tag, fmax_cutoff_option=True):
        """
        Summary:
            Function to generate .db file from ab-initio vasprun.xml file
            Could be used to add any keys for the new db
        Parameters:
            vasp_path: folder directory of multiple vasp calculation, run the script outside each single task folder
            fmax_cutoff: determine if an atomistic structure is relaxed
            fmax_cutoff_option: determine if apply relaxed structures
            zeo_tag: the tag for calling the whole db data
        Returns: None
        """
        new_db = connect(self.name)  # need to be new db's name
        _, dir_list, _ = next(os.walk(self.path))  # generator, need to be path to dft submission folder

        for dir_name in dir_list:
            traj = io.read(dir_name + '/vasprun.xml', ':')
            atoms_end = traj[-1]  # Atoms object, optimized structure of each xml file
            fmax_end = max(np.linalg.norm(atoms_end.get_forces(), axis=1))

            if fmax_cutoff_option:
                if fmax_end <= fmax_cutoff:
                    for atoms in traj:
                        fmax = max(np.linalg.norm(atoms.get_forces(), axis=1))
                        if fmax <= fmax_cutoff:
                            tag_relaxed = True
                        else:
                            tag_relaxed = False

                        new_db.write(atoms, relaxed=tag_relaxed, key=zeo_tag)

    def db_selector(self, **kwargs):
        """
        Summary:
            Extract traj list of atom configurations from database
        Parameters:
            kwargs: dictionary of db search key words
        Returns:
            traj: atoms trajectory list
        """
        traj = []
        db = connect(self.path + self.name)
        dict_v = [v for k, v in kwargs.items()]
        dict_v = ", ".join(dict_v)  # list to string
        for row in db.select(dict_v):
            atoms = db.get_atoms(id=row.id)
            traj.append(atoms)
        return traj

    @staticmethod
    def train_test_split(traj, training_weight, shuffle_mode=True):
        """
        Summary:
                Split selected trajectories into train and test sets
        Parameters:
            traj: traj file as a list from db_selector function
            training_weight: weight percentage of train-test split
            shuffle_mode: = 0 or 1 randomly selection or dummy selection
        Returns:
            length of traj file, training traj, valid traj
        """
        if shuffle_mode:  # shuffle the dataset
            shuffle(traj)
        else:
            pass

        len_traj = len(traj)
        len_train = round(training_weight*len(traj))
        traj_train = traj[:len_train]
        traj_valid = traj[len_train:]
        return len_traj, traj_train, traj_valid

    def db_saver(self, dft_dict, amp_dict, ):

        return
