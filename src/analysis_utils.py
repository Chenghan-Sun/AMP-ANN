# AMP analysis
from amp.descriptor.analysis import FingerprintPlot
from amp.analysis import plot_parity
from amp.analysis import plot_sensitivity
from amp.analysis import plot_convergence

# Visual
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from ase.visualize import view

class ModelAnalysis:
    """ implemented convergence and parameters analysis for Atomistic Machine Learning task
    """
    def __init__(self, default, logpath, logfile='./amp-log.txt'):
        self.logfile = logfile
        self.default = default
        self.logpath = logpath

    def conv_plot_default(self, label):
        """ convergence plot using default module
            Inputs: label (e.g. NN architecture 5x5)
            Outputs: .png convergence behavior plot
        """
        if self.default is True:
            logfile = self.logpath+'/'+self.logfile
            plot_convergence(logfile, plotfile='convergence.png')
        elif self.default is False:
            logfile = self.logpath+'/amp-log-'+label+'.txt'
            plot_convergence(logfile, plotfile='convergence-'+label+'.png')

    @staticmethod
    def extract_logfile(logfile):
        """ Reads the log file from the training process, returning the relevant
        parameters
            Inputs: amp-log.txt file from a training circle
            Outputs:
        """
        data = {}

        with open(logfile, 'r') as f:
            lines = f.read().splitlines()
            print(len(lines))
        print('file opened')

        """
        TODO: temporarily for QE Results:
        """
