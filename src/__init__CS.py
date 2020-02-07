import sys
import numpy as np
from ase.calculators.calculator import Parameters
from ..utilities import (Logger, ConvergenceOccurred, make_sublists, now,
                         setup_parallel)
try:
    from .. import fmodules
except ImportError:
    fmodules = None


class Model(object):
    """Class that includes common methods between different models."""

    @property
    def log(self):
        """Method to set or get a logger. Should be an instance of
        amp.utilities.Logger.

        Parameters
        ----------
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        """
        if hasattr(self, '_log'):
            return self._log
        if hasattr(self.parent, 'log'):
            return self.parent.log
        return Logger(None)

    @log.setter
    def log(self, log):
        self._log = log

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to re-establish the calculator."""
        # Make sure numpy prints out enough data.
        np.set_printoptions(precision=30, threshold=999999999)
        return self.parameters.tostring()

    def calculate_energy(self, fingerprints):
        """Calculates the model-predicted energy for an image, based on its
        fingerprint.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            self.atomic_energies = []
            energy = 0.0
            for index, (symbol, afp) in enumerate(fingerprints):
                atom_energy = self.calculate_atomic_energy(afp=afp,
                                                           index=index,
                                                           symbol=symbol)
                self.atomic_energies.append(atom_energy)
                energy += atom_energy
        return energy

    def calculate_forces(self, fingerprints, fingerprintprimes):
        """Calculates the model-predicted forces for an image, based on
        derivatives of fingerprints.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))
            for key in fingerprintprimes.keys():
                selfindex, selfsymbol, nindex, nsymbol, i = key
                derafp = fingerprintprimes[key]
                afp = fingerprints[nindex][1]
                dforce = self.calculate_force(afp=afp,
                                              derafp=derafp,
                                              nindex=nindex,
                                              nsymbol=nsymbol,
                                              direction=i,)
                forces[selfindex][i] += dforce
        return forces

    def calculate_dEnergy_dParameters(self, fingerprints):
        """Calculates a list of floats corresponding to the derivative of
        model-predicted energy of an image with respect to model parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            denergy_dparameters = None
            for index, (symbol, afp) in enumerate(fingerprints):
                temp = self.calculate_dAtomicEnergy_dParameters(afp=afp,
                                                                index=index,
                                                                symbol=symbol)
                if denergy_dparameters is None:
                    denergy_dparameters = temp
                else:
                    denergy_dparameters += temp
        return denergy_dparameters

    def calculate_numerical_dEnergy_dParameters(self, fingerprints, d=0.00001):
        """Evaluates dEnergy_dParameters using finite difference.

        This will trigger two calls to calculate_energy(), with each parameter
        perturbed plus/minus d.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            vector = self.vector
            denergy_dparameters = []
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                eplus = self.calculate_energy(fingerprints)
                vector[_] -= 2 * d
                self.vector = vector
                eminus = self.calculate_energy(fingerprints)
                denergy_dparameters += [(eplus - eminus) / (2 * d)]
                vector[_] += d
                self.vector = vector
            denergy_dparameters = np.array(denergy_dparameters)
        return denergy_dparameters

    def calculate_dForces_dParameters(self, fingerprints, fingerprintprimes):
        """Calculates an array of floats corresponding to the derivative of
        model-predicted atomic forces of an image with respect to model
        parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): None
                                   for selfindex in selfindices
                                   for i in range(3)}
            for key in fingerprintprimes.keys():
                selfindex, selfsymbol, nindex, nsymbol, i = key
                derafp = fingerprintprimes[key]
                afp = fingerprints[nindex][1]
                temp = self.calculate_dForce_dParameters(afp=afp,
                                                         derafp=derafp,
                                                         direction=i,
                                                         nindex=nindex,
                                                         nsymbol=nsymbol,)
                if dforces_dparameters[(selfindex, i)] is None:
                    dforces_dparameters[(selfindex, i)] = temp
                else:
                    dforces_dparameters[(selfindex, i)] += temp
        return dforces_dparameters

    def calculate_numerical_dForces_dParameters(self, fingerprints,
                                                fingerprintprimes, d=0.00001):
        """Evaluates dForces_dParameters using finite difference. This will
        trigger two calls to calculate_forces(), with each parameter perturbed
        plus/minus d.

        Parameters
        ---------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): []
                                   for selfindex in selfindices
                                   for i in range(3)}
            vector = self.vector
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                fplus = self.calculate_forces(fingerprints, fingerprintprimes)
                vector[_] -= 2 * d
                self.vector = vector
                fminus = self.calculate_forces(fingerprints, fingerprintprimes)
                for selfindex in selfindices:
                    for i in range(3):
                        dforces_dparameters[(selfindex, i)] += \
                            [(fplus[selfindex][i] - fminus[selfindex][i]) / (
                                2 * d)]
                vector[_] += d
                self.vector = vector
            for selfindex in selfindices:
                for i in range(3):
                    dforces_dparameters[(selfindex, i)] = \
                        np.array(dforces_dparameters[(selfindex, i)])
        return dforces_dparameters


class LossFunction:

    """Basic loss function, which can be used by the model.get_loss
    method which is required in standard model classes.

    This version is pure python and thus will be slow compared to a
    fortran/parallel implementation.

    If parallel is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.

    See self.default_parameters for the default values of parameters
    specified as None.

    Parameters
    ----------
    energy_coefficient : float
        Coefficient of the energy contribution in the loss function.
    force_coefficient : float
        Coefficient of the force contribution in the loss function.
        Can set to None as shortcut to turn off force training.
    convergence : dict
        Dictionary of keys and values defining convergence.  Keys are
        'energy_rmse', 'energy_maxresid', 'force_rmse', and 'force_maxresid'.
        If 'force_rmse' and 'force_maxresid' are both set to None, force
        training is turned off and force_coefficient is set to None.
    parallel : dict
        Parallel configuration dictionary. Will pull from model itself if
        not specified.
    overfit : float
        Multiplier of the weights norm penalty term in the loss function.
    raise_ConvergenceOccurred : bool
        If True will raise convergence notice.
    log_losses : bool
        If True will log the loss function value in the log file else will not.
    d : None or float
        If d is None, both loss function and its gradient are calculated
        analytically. If d is a float, then gradient of the loss function is
        calculated by perturbing each parameter plus/minus d.
    """

    default_parameters = {'convergence': {'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': None,
                                          'force_maxresid': None, }
                          }

    def __init__(self, indices_fit_forces='all', energy_coefficient=1.0, force_coefficient=0.04,  # CS
                 convergence=None, parallel=None, overfit=0.,
                 raise_ConvergenceOccurred=True, log_losses=True, d=None):
        self.indices_fit_forces = indices_fit_forces
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        # 'dict' creates a copy; otherwise mutable in class.
        c = p['convergence'] = dict(self.default_parameters['convergence'])
        if convergence is not None:
            for key, value in convergence.items():
                p['convergence'][key] = value
        p['energy_coefficient'] = energy_coefficient
        p['force_coefficient'] = force_coefficient
        p['overfit'] = overfit
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self.log_losses = log_losses
        self.d = d
        self._step = 0
        self._initialized = False
        self._data_sent = False
        self._parallel = parallel
        if (c['force_rmse'] is None) and (c['force_maxresid'] is None):
            p['force_coefficient'] = None
        if p['force_coefficient'] is None:
            c['force_rmse'] = None
            c['force_maxresid'] = None

    def attach_model(self, model, fingerprints=None,
                     fingerprintprimes=None, images=None):
        """Attach the model to be used to the loss function.

        fingerprints and training images need not be supplied if they are
        already attached to the model via model.trainingparameters.

        Parameters
        ----------
        model : object
            Class representing the regression model.
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the training set of data. This can
            also be the path to an ASE trajectory (.traj) or database (.db)
            file. Energies can be obtained from any reference, e.g. DFT
            calculations.
        """
        self._model = model
        self.fingerprints = fingerprints
        self.fingerprintprimes = fingerprintprimes
        self.images = images

    def _initialize(self):
        """Procedures to be run on the first call only, such as establishing
        SSH sessions, etc."""
        if self._initialized is True:
            return

        if self._parallel is None:
            self._parallel = self._model._parallel
        log = self._model.log

        if self.fingerprints is None:
            self.fingerprints = \
                self._model.trainingparameters.descriptor.fingerprints

        # May also make sense to decide whether or not to calculate
        # fingerprintprimes based on the value of train_forces.
        if ((self.parameters.force_coefficient is not None) and
                (self.fingerprintprimes is None)):
            self.fingerprintprimes = \
                self._model.trainingparameters.descriptor.fingerprintprimes
        if self.images is None:
            self.images = self._model.trainingparameters.trainingimages

        if self._parallel['cores'] != 1:  # Initialize workers.
            python = sys.executable
            workercommand = '%s -m %s' % (python, self.__module__)
            server, connections, n_pids = setup_parallel(self._parallel,
                                                         workercommand, log)
            self._sessions = {'master': server,
                              'connections': connections,  # SSH's/nodes
                              'n_pids': n_pids}  # total no. of workers

        if self.log_losses:
            p = self.parameters
            convergence = p['convergence']
            log(' Loss function convergence criteria:')
            log('  energy_rmse: ' + str(convergence['energy_rmse']))
            log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
            log('  force_rmse: ' + str(convergence['force_rmse']))
            log('  force_maxresid: ' + str(convergence['force_maxresid']))
            log(' Loss function set-up:')
            log('  energy_coefficient: ' + str(p.energy_coefficient))
            log('  force_coefficient: ' + str(p.force_coefficient))
            log('  overfit: ' + str(p.overfit))
            # log(' Indices to fit (CS): ' + str(p.indices_fit_forces) # CS 
            log('\n')
            
            if p.force_coefficient is None:
                header = '%5s %19s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12))
            else:
                header = '%5s %19s %12s %12s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy',
                     '', 'Force'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid',
                     'ForceRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12,
                     '=' * 12, '=' * 12))

        self._initialized = True

    def _send_data_to_fortran(self,):
        """Procedures to be run in fortran mode for a single requested core
        only. Also just on the first call for sending data to fortran modules.
        """
        if self._data_sent is True:
            return

        num_images = len(self.images)
        p = self.parameters
        energy_coefficient = p.energy_coefficient
        overfit = p.overfit
        if p.force_coefficient is None:
            train_forces = False
            force_coefficient = 0.
        else:
            train_forces = True
            force_coefficient = p.force_coefficient
        mode = self._model.parameters.mode
        if mode == 'atom-centered':
            num_atoms = None
        elif mode == 'image-centered':
            raise NotImplementedError('Image-centered mode is not coded yet.')

        (actual_energies, actual_forces, elements, atomic_positions,
         num_images_atoms, atomic_numbers, raveled_fingerprints, num_neighbors,
         raveled_neighborlists, raveled_fingerprintprimes) = (None,) * 10

        value = ravel_data(train_forces,
                           mode,
                           self.images,
                           self.fingerprints,
                           self.fingerprintprimes,)

        if mode == 'image-centered':
            if not train_forces:
                (actual_energies, atomic_positions) = value
            else:
                (actual_energies, actual_forces, atomic_positions) = value
        else:
            if not train_forces:
                (actual_energies, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints) = value
            else:
                (actual_energies, actual_forces, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints, num_neighbors,
                 raveled_neighborlists, raveled_fingerprintprimes) = value

        send_data_to_fortran(fmodules,
                             energy_coefficient,
                             force_coefficient,
                             overfit,
                             train_forces,
                             num_atoms,
                             num_images,
                             actual_energies,
                             actual_forces,
                             atomic_positions,
                             num_images_atoms,
                             atomic_numbers,
                             raveled_fingerprints,
                             num_neighbors,
                             raveled_neighborlists,
                             raveled_fingerprintprimes,
                             self._model,
                             self.d)
        self._data_sent = True

    def _cleanup(self):
        """Closes SSH sessions."""
        self._initialized = False
        if not hasattr(self, '_sessions'):
            return
        server = self._sessions['master']

        finished = np.array([False] * self._sessions['n_pids'])
        while not finished.all():
            message = server.recv_pyobj()
            if (message['subject'] == '<request>' and
                    message['data'] == 'parameters'):
                server.send_pyobj('<stop>')
                finished[int(message['id'])] = True

        for _ in self._sessions['connections']:
            if hasattr(_, 'logout'):
                _.logout()
        del self._sessions['connections']

    def get_loss(self, parametervector, lossprime):
        """Returns the current value of the loss function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.
        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """

        self._initialize()

        if self._parallel['cores'] == 1:
            if self._model.fortran:
                self._model.vector = parametervector
                self._send_data_to_fortran()
                (loss, dloss_dparameters, energy_loss, force_loss,
                 energy_maxresid, force_maxresid) = \
                    fmodules.calculate_loss(parameters=parametervector,
                                            num_parameters=len(
                                                parametervector),
                                            lossprime=lossprime)
            else:
                loss, dloss_dparameters, energy_loss, force_loss, \
                    energy_maxresid, force_maxresid = \
                    self.calculate_loss(parametervector,
                                        lossprime=lossprime)
        else:
            server = self._sessions['master']
            n_pids = self._sessions['n_pids']

            # Subdivide tasks.
            keys = make_sublists(self.images.keys(), n_pids)

            args = {'lossprime': lossprime,
                    'd': self.d}

            results = self.process_parallels(parametervector,
                                             server,
                                             n_pids,
                                             keys,
                                             args=args)
            loss = results['loss']
            dloss_dparameters = results['dloss_dparameters']
            energy_loss = results['energy_loss']
            force_loss = results['force_loss']
            energy_maxresid = results['energy_maxresid']
            force_maxresid = results['force_maxresid']

        self.loss, self.energy_loss, self.force_loss, \
            self.energy_maxresid, self.force_maxresid = \
            loss, energy_loss, force_loss, energy_maxresid, force_maxresid

        if lossprime:
            self.dloss_dparameters = dloss_dparameters

        if self.raise_ConvergenceOccurred:
            # Only during calculation of loss function (and not lossprime)
            # convergence is checked and values are printed out in the log
            # file.
            if lossprime is False:
                self._model.vector = parametervector
                converged = self.check_convergence(loss,
                                                   energy_loss,
                                                   force_loss,
                                                   energy_maxresid,
                                                   force_maxresid)
                if converged:
                    self._cleanup()
                    if self._parallel['cores'] != 1:
                        # Needed to properly close socket connection
                        # (python3).
                        server.close()
                    raise ConvergenceOccurred()

        return {'loss': self.loss,
                'dloss_dparameters': (self.dloss_dparameters
                                      if lossprime is True
                                      else dloss_dparameters),
                'energy_loss': self.energy_loss,
                'force_loss': self.force_loss,
                'energy_maxresid': self.energy_maxresid,
                'force_maxresid': self.force_maxresid, }

    def calculate_loss(self, parametervector, lossprime):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.

        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """
        self._model.vector = parametervector
        p = self.parameters
        energyloss = 0.
        forceloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        dloss_dparameters = np.array([0.] * len(parametervector))
        model = self._model
        for hash in self.images.keys():
            image = self.images[hash]
            no_of_atoms = len(image)
            amp_energy = model.calculate_energy(self.fingerprints[hash])
            actual_energy = image.get_potential_energy(apply_constraint=False)
            residual_per_atom = abs(amp_energy - actual_energy) / \
                len(image)
            if residual_per_atom > energy_maxresid:
                energy_maxresid = residual_per_atom
            energyloss += residual_per_atom**2

            # Calculates derivative of the loss function with respect to
            # parameters if lossprime is true
            if lossprime:
                if model.parameters.mode == 'image-centered':
                    raise NotImplementedError('This needs to be coded.')
                elif model.parameters.mode == 'atom-centered':
                    if self.d is None:
                        denergy_dparameters = \
                            model.calculate_dEnergy_dParameters(
                                self.fingerprints[hash])
                    else:
                        denergy_dparameters = \
                            model.calculate_numerical_dEnergy_dParameters(
                                self.fingerprints[hash], d=self.d)
                    temp = p.energy_coefficient * 2. * \
                        (amp_energy - actual_energy) * \
                        denergy_dparameters / \
                        (no_of_atoms ** 2.)
                    dloss_dparameters += temp

            if p.force_coefficient is not None:
                amp_forces = \
                    model.calculate_forces(self.fingerprints[hash],
                                           self.fingerprintprimes[hash])
                actual_forces = image.get_forces(apply_constraint=False)
                if self.indices_fit_forces == 'all': # CS 
                	self.indices_fit_forces = range(no_of_atoms) # CS
                else: # CS
                    print(f'Selected atoms for training are {self.indices_fit_forces}') # CS
                for index in self.indices_fit_forces: #range(no_of_atoms): # CS
                    for i in range(3):
                        force_resid = abs(amp_forces[index][i] -
                                          actual_forces[index][i])
                        if force_resid > force_maxresid:
                            force_maxresid = force_resid
                        temp = (1. / 3.) * (amp_forces[index][i] -
                                            actual_forces[index][i]) ** 2. / \
                            no_of_atoms
                        forceloss += temp
                # Calculates derivative of the loss function with respect to
                # parameters if lossprime is true
                if lossprime:
                    if model.parameters.mode == 'image-centered':
                        raise NotImplementedError('This needs to be coded.')
                    elif model.parameters.mode == 'atom-centered':
                        if self.d is None:
                            dforces_dparameters = \
                                model.calculate_dForces_dParameters(
                                    self.fingerprints[hash],
                                    self.fingerprintprimes[hash])
                        else:
                            dforces_dparameters = \
                                model.calculate_numerical_dForces_dParameters(
                                    self.fingerprints[hash],
                                    self.fingerprintprimes[hash],
                                    d=self.d)
                        for selfindex in range(no_of_atoms):
                            for i in range(3):
                                temp = p.force_coefficient * (2.0 / 3.0) * \
                                    (amp_forces[selfindex][i] -
                                     actual_forces[selfindex][i]) * \
                                    dforces_dparameters[(selfindex, i)] \
                                    / no_of_atoms
                                dloss_dparameters += temp

        loss = p.energy_coefficient * energyloss
        if p.force_coefficient is not None:
            loss += p.force_coefficient * forceloss
        dloss_dparameters = np.array(dloss_dparameters)

        # if overfit coefficient is more than zero, overfit contribution to
        # loss and dloss_dparameters is also added.
        if p.overfit > 0.:
            overfitloss = 0.
            for component in parametervector:
                overfitloss += component ** 2.
            overfitloss *= p.overfit
            loss += overfitloss
            doverfitloss_dparameters = \
                2 * p.overfit * np.array(parametervector)
            dloss_dparameters += doverfitloss_dparameters

        return loss, dloss_dparameters, energyloss, forceloss, \
            energy_maxresid, force_maxresid

    # All incoming requests will be dictionaries with three keys.
    # d['id']: process id number, assigned when process created above.
    # d['subject']: what the message is asking for / telling you.
    # d['data']: optional data passed from worker.

    def process_parallels(self, vector, server, n_pids, keys, args):
        """

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        server : object
            Master session of parallel processing.
        processes: list of objects
            Worker sessions for parallel processing.
        keys : list
            List of images keys for worker processes.
        args : dict
            Dictionary containing arguments of the method to be called on each
            worker process.
        """
        # For each process
        finished = np.array([False] * n_pids)
        results = {'loss': 0.,
                   'dloss_dparameters': [0.] * len(vector),
                   'energy_loss': 0.,
                   'force_loss': 0.,
                   'energy_maxresid': 0.,
                   'force_maxresid': 0.}
        while not finished.all():
            message = server.recv_pyobj()
            if message['subject'] == '<purpose>':
                server.send_string('calculate_loss_function')
            elif message['subject'] == '<request>':
                request = message['data']  # Variable name.
                if request == 'images':
                    subimages = {k: self.images[k] for k in
                                 keys[int(message['id'])]}
                    server.send_pyobj(subimages)
                elif request == 'fortran':
                    server.send_pyobj(self._model.fortran)
                elif request == 'modelstring':
                    server.send_pyobj(self._model.tostring())
                elif request == 'lossfunctionstring':
                    server.send_pyobj(self.parameters.tostring())
                elif request == 'fingerprints':
                    server.send_pyobj({k: self.fingerprints[k] for k in
                                       keys[int(message['id'])]})
                elif request == 'fingerprintprimes':
                    if self.fingerprintprimes is not None:
                        server.send_pyobj({k: self.fingerprintprimes[k] for k
                                           in keys[int(message['id'])]})
                    else:
                        server.send_pyobj(None)
                elif request == 'args':
                    server.send_pyobj(args)
                elif request == 'parameters':
                    if finished[int(message['id'])]:
                        server.send_pyobj('<continue>')
                    else:
                        server.send_pyobj(vector)
                else:
                    raise NotImplementedError()
            elif message['subject'] == '<result>':
                result = message['data']
                server.send_string('meaningless reply')

                results['loss'] += result['loss']
                results['dloss_dparameters'] += result['dloss_dparameters']
                results['energy_loss'] += result['energy_loss']
                results['force_loss'] += result['force_loss']
                if result['energy_maxresid'] > results['energy_maxresid']:
                    results['energy_maxresid'] = result['energy_maxresid']
                if result['force_maxresid'] > results['force_maxresid']:
                    results['force_maxresid'] = result['force_maxresid']
                finished[int(message['id'])] = True

        return results

    def check_convergence(self, loss, energy_loss, force_loss,
                          energy_maxresid, force_maxresid):
        """Check convergence

        Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer.

        Parameters
        ----------
        loss : float
            Value of the loss function.
        energy_loss : float
            Value of the energy contribution of the loss function.
        force_loss : float
            Value of the force contribution of the loss function.
        energy_maxresid : float
            Maximum energy residual.
        force_maxresid : float
            Maximum force residual.
        """
        p = self.parameters
        energy_rmse_converged = True
        log = self._model.log
        if p.convergence['energy_rmse'] is not None:
            energy_rmse = np.sqrt(energy_loss / len(self.images))
            if energy_rmse > p.convergence['energy_rmse']:
                energy_rmse_converged = False
        energy_maxresid_converged = True
        if p.convergence['energy_maxresid'] is not None:
            if energy_maxresid > p.convergence['energy_maxresid']:
                energy_maxresid_converged = False
        if p.force_coefficient is not None:
            force_rmse_converged = True
            if p.convergence['force_rmse'] is not None:
                force_rmse = np.sqrt(force_loss / len(self.images))
                if force_rmse > p.convergence['force_rmse']:
                    force_rmse_converged = False
            force_maxresid_converged = True
            if p.convergence['force_maxresid'] is not None:
                if force_maxresid > p.convergence['force_maxresid']:
                    force_maxresid_converged = False

            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s'
                    ' %10.4e %1s %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-',
                     force_rmse,
                     'C' if force_rmse_converged else '-',
                     force_maxresid,
                     'C' if force_maxresid_converged else '-'))

            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged and \
                force_rmse_converged and force_maxresid_converged
        else:
            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-'))
            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged


def calculate_fingerprints_range(fp, images):
    """Calculates the range for the fingerprints corresponding to images,
    stored in fp. fp is a fingerprints object with the fingerprints data
    stored in a dictionary-like object at fp.fingerprints. (Typically this
    is a .utilties.Data structure.) images is a hashed dictionary of atoms
    for which to consider the range.

    In image-centered mode, returns an array of (min, max) values for each
    fingerprint. In atom-centered mode, returns a dictionary of such
    arrays, one per element.
    """
    if fp.parameters.mode == 'image-centered':
        raise NotImplementedError()
    elif fp.parameters.mode == 'atom-centered':
        fprange = {}
        for hash in images.keys():
            imagefingerprints = fp.fingerprints[hash]
            for element, fingerprint in imagefingerprints:
                if element not in fprange:
                    fprange[element] = [[_, _] for _ in fingerprint]
                else:
                    assert len(fprange[element]) == len(fingerprint)
                    for i, ridge in enumerate(fingerprint):
                        if ridge < fprange[element][i][0]:
                            fprange[element][i][0] = ridge
                        elif ridge > fprange[element][i][1]:
                            fprange[element][i][1] = ridge
    for key, value in fprange.items():
        fprange[key] = value
    return fprange


def ravel_data(train_forces,
               mode,
               images,
               fingerprints,
               fingerprintprimes,):
    """
    Reshapes data of images into lists.

    Parameters
    ---------
    train_forces : bool
        Determining whether forces are also trained or not.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This is the training set of data. This can also be the
        path to an ASE trajectory (.traj) or database (.db) file. Energies can
        be obtained from any reference, e.g. DFT calculations.

    fingerprints : dict
        Dictionary with images hashs as keys and the corresponding fingerprints
        as values.
    fingerprintprimes : dict
        Dictionary with images hashs as keys and the corresponding fingerprint
        derivatives as values.
    """
    from ase.data import atomic_numbers

    actual_energies = [image.get_potential_energy(apply_constraint=False)
                       for image in images.values()]

    if mode == 'atom-centered':
        num_images_atoms = [len(image) for image in images.values()]
        atomic_numbers = [atomic_numbers[atom.symbol]
                          for image in images.values() for atom in image]

        def ravel_fingerprints(images,
                               fingerprints):
            """
            Reshape fingerprints of images into a list.
            """
            raveled_fingerprints = []
            elements = []
            for hash, image in images.items():
                for index in range(len(image)):
                    elements += [fingerprints[hash][index][0]]
                    raveled_fingerprints += [fingerprints[hash][index][1]]
            elements = sorted(set(elements))
            # Could also work without images:
#            raveled_fingerprints = [afp
#                    for hash, value in fingerprints.iteritems()
#                    for (element, afp) in value]
            return elements, raveled_fingerprints

        elements, raveled_fingerprints = ravel_fingerprints(images,
                                                            fingerprints)
    else:
        atomic_positions = [image.positions.ravel()
                            for image in images.values()]

    if train_forces is True:

        actual_forces = \
            [image.get_forces(apply_constraint=False)[index]
             for image in images.values() for index in range(len(image))]

        if mode == 'atom-centered':

            def ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes):
                """
                Reshape neighborlists and fingerprintprimes of images into a
                list and a matrix, respectively.
                """
                # Only neighboring atoms of type II (within the main cell)
                # need to be sent to fortran for force training.
                # All keys in fingerprintprimes are for type II neighborhoods.
                # Also note that each atom is considered as neighbor of
                # itself in fingerprintprimes.
                num_neighbors = []
                raveled_neighborlists = []
                raveled_fingerprintprimes = []
                for hash, image in images.items():
                    for atom in image:
                        selfindex = atom.index
                        selfsymbol = atom.symbol
                        selfneighborindices = []
                        selfneighborsymbols = []
                        for key, derafp in fingerprintprimes[hash].items():
                            # key = (selfindex, selfsymbol, nindex, nsymbol, i)
                            # i runs from 0 to 2. neighbor indices and symbols
                            # should be added just once.
                            if key[0] == selfindex and key[4] == 0:
                                selfneighborindices += [key[2]]
                                selfneighborsymbols += [key[3]]

                        neighborcount = 0
                        for nindex, nsymbol in zip(selfneighborindices,
                                                   selfneighborsymbols):
                            raveled_neighborlists += [nindex]
                            neighborcount += 1
                            for i in range(3):
                                fpprime = fingerprintprimes[hash][(selfindex,
                                                                   selfsymbol,
                                                                   nindex,
                                                                   nsymbol,
                                                                   i)]
                                raveled_fingerprintprimes += [fpprime]
                        num_neighbors += [neighborcount]

                return (num_neighbors,
                        raveled_neighborlists,
                        raveled_fingerprintprimes)

            (num_neighbors,
             raveled_neighborlists,
             raveled_fingerprintprimes) = \
                ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes)
    if mode == 'image-centered':
        if not train_forces:
            return (actual_energies, atomic_positions)
        else:
            return (actual_energies, actual_forces, atomic_positions)
    else:
        if not train_forces:
            return (actual_energies, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints)
        else:
            return (actual_energies, actual_forces, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints, num_neighbors,
                    raveled_neighborlists, raveled_fingerprintprimes)


def send_data_to_fortran(_fmodules,
                         energy_coefficient,
                         force_coefficient,
                         overfit,
                         train_forces,
                         num_atoms,
                         num_images,
                         actual_energies,
                         actual_forces,
                         atomic_positions,
                         num_images_atoms,
                         atomic_numbers,
                         raveled_fingerprints,
                         num_neighbors,
                         raveled_neighborlists,
                         raveled_fingerprintprimes,
                         model,
                         d):
    """
    Function that sends images data to fortran code. Is used just once on each
    core.
    """
    from ase.data import atomic_numbers as an

    if model.parameters.mode == 'image-centered':
        mode_signal = 1
    elif model.parameters.mode == 'atom-centered':
        mode_signal = 2

    _fmodules.images_props.num_images = num_images
    _fmodules.images_props.actual_energies = actual_energies
    if train_forces:
        _fmodules.images_props.actual_forces = actual_forces

    _fmodules.model_props.energy_coefficient = energy_coefficient
    _fmodules.model_props.force_coefficient = force_coefficient
    _fmodules.model_props.overfit = overfit
    _fmodules.model_props.train_forces = train_forces
    _fmodules.model_props.mode_signal = mode_signal
    if d is None:
        _fmodules.model_props.numericprime = False
    else:
        _fmodules.model_props.numericprime = True
        _fmodules.model_props.d = d

    if model.parameters.mode == 'atom-centered':
        fprange = model.parameters.fprange
        elements = sorted(fprange.keys())
        num_elements = len(elements)
        elements_numbers = [an[elm] for elm in elements]
        min_fingerprints = \
            [[fprange[elm][_][0] for _ in range(len(fprange[elm]))]
             for elm in elements]
        max_fingerprints = [[fprange[elm][_][1]
                             for _
                             in range(len(fprange[elm]))]
                            for elm in elements]
        num_fingerprints_of_elements = \
            [len(fprange[elm]) for elm in elements]

        _fmodules.images_props.num_elements = num_elements
        _fmodules.images_props.elements_numbers = elements_numbers
        _fmodules.images_props.num_images_atoms = num_images_atoms
        _fmodules.images_props.atomic_numbers = atomic_numbers
        if train_forces:
            _fmodules.images_props.num_neighbors = num_neighbors
            _fmodules.images_props.raveled_neighborlists = \
                raveled_neighborlists

        _fmodules.fingerprint_props.num_fingerprints_of_elements = \
            num_fingerprints_of_elements
        _fmodules.fingerprint_props.raveled_fingerprints = raveled_fingerprints
        _fmodules.neuralnetwork.min_fingerprints = min_fingerprints
        _fmodules.neuralnetwork.max_fingerprints = max_fingerprints
        if train_forces:
            _fmodules.fingerprint_props.raveled_fingerprintprimes = \
                raveled_fingerprintprimes
    else:
        _fmodules.images_props.num_atoms = num_atoms
        _fmodules.images_props.atomic_positions = atomic_positions

    # for neural neyworks only
    if model.parameters['importname'] == '.model.neuralnetwork.NeuralNetwork':

        hiddenlayers = model.parameters.hiddenlayers
        activation = model.parameters.activation

        if model.parameters.mode == 'atom-centered':
            from collections import OrderedDict
            no_layers_of_elements = \
                [3 if isinstance(hiddenlayers[elm], int)
                 else (len(hiddenlayers[elm]) + 2)
                 for elm in elements]
            nn_structure = OrderedDict()
            for elm in elements:
                len_of_fps = len(fprange[elm])
                if isinstance(hiddenlayers[elm], int):
                    nn_structure[elm] = \
                        ([len_of_fps] + [hiddenlayers[elm]] + [1])
                else:
                    nn_structure[elm] = \
                        ([len_of_fps] +
                         [layer for layer in hiddenlayers[elm]] + [1])

            no_nodes_of_elements = [nn_structure[elm][_]
                                    for elm in elements
                                    for _ in range(len(nn_structure[elm]))]

        else:
            num_atoms = model.parameters.num_atoms
            if isinstance(hiddenlayers, int):
                no_layers_of_elements = [3]
            else:
                no_layers_of_elements = [len(hiddenlayers) + 2]
            if isinstance(hiddenlayers, int):
                nn_structure = ([3 * num_atoms] + [hiddenlayers] + [1])
            else:
                nn_structure = ([3 * num_atoms] +
                                [layer for layer in hiddenlayers] + [1])
            no_nodes_of_elements = [nn_structure[_]
                                    for _ in range(len(nn_structure))]

        _fmodules.neuralnetwork.no_layers_of_elements = no_layers_of_elements
        _fmodules.neuralnetwork.no_nodes_of_elements = no_nodes_of_elements
        if activation == 'tanh':
            activation_signal = 1
        elif activation == 'sigmoid':
            activation_signal = 2
        elif activation == 'linear':
            activation_signal = 3
        _fmodules.neuralnetwork.activation_signal = activation_signal
