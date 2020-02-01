# Database Module
from ase.db import connect

# Math Module
from random import shuffle


class DatabaseTools:
    """
    Summary:
        ASE Database tools used for sampling data from raw DFT calculations
    init:
        specify directory path of database, database name
        default all training-validation .traj file names
        TODO: #1 add read-in vasprun.xml file functionality
    """
    def __init__(self, db_path, db_name, dataset='all.traj',
                 training_traj='trainset.traj', validation_traj='validset.traj'):
        self.path = db_path
        self.name = db_name
        self.dataset = dataset
        self.training_traj = training_traj
        self.validation_traj = validation_traj

    def xml_to_db(self, ):
        """
        Summary:
            Function to generate
        """

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
                split selected trajs into train and test sets
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

    def db_saver(self, ):
        """

        """
        return