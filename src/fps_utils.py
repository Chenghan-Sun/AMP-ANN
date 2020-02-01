"""
This File:

"""
import sys
import numpy as np

# AMP Module
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.utilities import hash_images

# Visual
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class FpsAnalysisTools:
    """ Summary:
            Some notes about the fp data structure:
            For 7nps@Zeo system
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
    def __init__(self, metal, mode, ):
        """ Parameters:
                metal: type of metal nanocluster
                mode:
                    =0: only the metal nanocluster
                    =1: netal nanocluster with zeolite framework
        """
        self.metal = metal
        self.mode = mode

    def descriptor_generator(self, G2_etas, G4_etas, G4_zetas, G4_gammas):
        """ Summary:
                Use this function solely for Amp calculator set-up
            Parameters:
                #1-4 Use Gaussian Functions as fingerprints kernel by changing
                the parameters of G2 and G4
            Returns:
                the descriptor: <amp.descriptor.gaussian.Gaussian object at 0x1013116a0>
                G data structure：
                example:
                    [{'type': 'G2', 'element': 'Pt', 'eta': 0.05}, {'type': 'G2', 'element': 'Pt', 'eta': 4.0},
                    {'type': 'G2', 'element': 'Pt', 'eta': 20.0}, {'type': 'G2', 'element': 'Pt', 'eta': 80.0},
                    {'type': 'G4', 'elements': ['Pt', 'Pt'], 'eta': 0.005, 'gamma': 1.0, 'zeta': 1.0},
                    {'type': 'G4', 'elements': ['Pt', 'Pt'], 'eta': 0.005, 'gamma': -1.0, 'zeta': 1.0},
                    {'type': 'G4', 'elements': ['Pt', 'Pt'], 'eta': 0.005, 'gamma': 1.0, 'zeta': 4.0},
                    {'type': 'G4', 'elements': ['Pt', 'Pt'], 'eta': 0.005, 'gamma': -1.0, 'zeta': 4.0}]
        """
        elements = [self.metal]
        if self.mode == 0:
            pass
        elif self.mode == 1:
            elements.append('Si')
            elements.append('O')
        else:
            raise Exception('mode should be 0 or 1. The value of mode was: {}'.format(self.mode))

        G = make_symmetry_functions(elements=elements, type='G2',
                                    etas=G2_etas)
        G += make_symmetry_functions(elements=elements, type='G4',
                                     etas=G4_etas,
                                     zetas=G4_zetas,
                                     gammas=G4_gammas)
        my_Gs = {self.metal: G}
        if self.mode == 0:
            pass
        elif self.mode == 1:
            my_Gs.update({'Si': G, 'O': G})
        descriptor = Gaussian(Gs=my_Gs)
        return descriptor, G

    @staticmethod
    def calc_fp(descriptor, images):
        """ Summary:
                transform .traj images to hash images
            Parameters:
                descriptor: from fcn --> descriptor_generator
                images: by ase.io.read() .traj file
            Returns:
                descriptor: descriptor with calculated fps
                images: hashed images
        """
        images = hash_images(images)
        descriptor.calculate_fingerprints(images)
        return descriptor, images

    def fp_values_extractor(self, atom_index, G, hash, descriptor, fig_name="", fig_flag=False):
        """ Summary:
                Extract the fingerprint value from hased images based on descriptor
                a hash = a image = an atomistic system
            Parameters:
                atom_index: fp of which number of atom in the system
                    e.g. atom_index=0 is the first O element
                G: symmetry functions, return from fcn --> descriptor_generator
                hash: receive the arg from emurating the images from fcn --> calc_fp
                descriptor from fcn --> calc_fp
                figure name: given name of the plotfile, default to ""
                fig_flag: falg for if user wants a plot
            Returns:
                printout information about the fingerprints of a atom in a hash image
                scatter plot of the fingerprint values
        """
        fp = descriptor.fingerprints[hash][atom_index][1] # unpack the hash images
        for j, val in enumerate(fp):
            g = G[j]
            if g['type'] == 'G2':
                ele = g['element']
            elif g['type'] == 'G4':
                ele = g['elements']
            else:
                raise Exception("No such symmetry function implemented")
            print(j, g['type'], ele, g['eta'], val)

            if fig_flag == True:
                if self.metal in ele:
                    my_marker = 'o' # o refers element with metal
                else:
                    my_marker = 'x'
                plt.plot(j, val, my_marker)
                plt.savefig(fig_name)

    @staticmethod
    def get_extreme_value_fp(descriptor, images):
        """ Summary:
                Extract the maximum and minimum values of all the fingerprints generated
            Parameters:
                enumerating the images returned by fcn --> calc_fp
            Returns:
                printout maximum and minimum values of fp
        """
        for index, hash in enumerate(images.keys()):
            fp = descriptor.fingerprints[hash]  # [0][1] all the information of fp
            max_all_image_fp_list = []  # max fp value of all images
            min_all_image_fp_list = []  # min fp value of all images
            for i in fp:
                # i = each atom and its 36 fp values
                # print(i[1]) #5(image)X43(row)X36(column)
                max_i_fp = max(i[1])  # i[0] --> atom type, i[1] --> 36 fp values
                min_i_fp = min(i[1])
                max_all_image_fp_list.append(max_i_fp)
                min_all_image_fp_list.append(min_i_fp)

            max_fp = max(max_all_image_fp_list)
            min_fp = min(min_all_image_fp_list)
            print("The maximum value of the fp is " + str(max_fp))
            print("The minimum value of the fp is " + str(min_fp))
        return max_fp, min_fp

    def fp_barplot(self, atom_index, G, hash, descriptor, fig_name, title):
        """ Summary:
                Makes a barplot of the fingerprints for a certain atom
            Parameters:
                atom_index: fp of which number of atom in the system
                    e.g. atom_index=0 is the first O element
                G: symmetry functions, return from fcn --> descriptor_generator
                hash: receive the arg from emurating the images from fcn --> calc_fp
                descriptor from fcn --> calc_fp
                figure name: given name of the plotfile
                title: title of each sub-polt
            Returns:
                barplot of the fingerprint components for a certain atom
        """
        fp = descriptor.fingerprints[hash][atom_index][1]  # list
        fig, ax = plt.subplots()
        G2_val = []
        G4_val = []
        metal_index_list = []  # lists indicate indices of zeo or metal
        zeo_index_list = []
        metal_ele_list = []  # lists indicate zeo or metal
        zeo_ele_list = []

        for j, val in enumerate(fp):
            g = G[j]
            if g['type'] == 'G2':
                G2_val.append(val)
                ele = g['element']
            elif g['type'] == 'G4':
                G4_val.append(val)
                ele = g['elements']

            if self.mode == 0:
                metal_index_list.append(j)
                metal_ele_list.append(val)
            elif self.mode == 1:
                if self.metal in ele:
                    metal_index_list.append(j)
                    metal_ele_list.append(val)
                else:
                    zeo_index_list.append(j)
                    zeo_ele_list.append(val)

        ax.bar(range(len(G2_val)), G2_val, label="G1")  # set the x-y axis
        ax.bar(np.arange(len(G4_val)) + len(G2_val), G4_val, label="G2")
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0., 12.)
        ax.set_xlabel('Index of fingerprints', fontsize=14)
        ax.set_ylabel('Value of fingerprints', fontsize=14)
        ax.scatter(metal_index_list, metal_ele_list, marker='o', color='purple', label='metal atom')
        if self.mode == 1:
            ax.scatter(zeo_index_list, zeo_ele_list, marker='x', color='purple', label='only zeo atoms')
        ax.legend(fontsize=14)
        fig.savefig(fig_name)

