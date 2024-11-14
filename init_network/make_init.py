
import numpy as np
import gsd
import gsd.hoomd
import sys, os
import sys, math, six
import scipy.spatial
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as KDTree
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng()


class RandomNetworkMaker():

    def __init__(self,L,num_backbone,length_backbone,crosslinking_probability,
                 length_crosslinkers,length_sidechains,azo_probability,num_ionic_liquid,
                 azo_architecture):
        ## define chemical tags ----
        self.particle_types = ['TC1','SN4a','SN3r','SC1','TN3r','TC5','TN2q','TC6','TC3','SX4e','SQ4n',]
        ## ---------------[  0  , 1  ,     2     ,  3 ,     4     ,   5]
        self.bond_types = ['TC1-TC1','TC1-SN4a','SN4a-SN3r','SN3r-SN3r',
                           'SN4a-SC1','SC1-SC1','SC1-TN3r','TN3r-TC5',
                           'TC5-TC5','SC1-TN2q',
                           'SX4e-SQ4n','SQ4n-SQ4n','TN2q-TN2q','TN2q-TC3','TN2q-TC6']
        self.angle_types = ['generic','backbone','PEO','azo_ring','azo_trans_isomer',
                            'azo_cis_isomer','TFSI1','TFSI2','EMIM']
        self.dihedral_types = ['PEO','TFSI']


        self.TC1 = 0
        self.SN4a = 1
        self.SN3r = 2
        self.SC1 = 3
        self.TN3r = 4
        self.TC5 = 5
        self.TN2q = 6
        self.TC6 = 7
        self.TC3 = 8
        self.SX4e = 9
        self.SQ4n = 10

        self.TC1_TC1 = 0
        self.TC1_SN4a = 1
        self.SN4a_SN3r = 2
        self.SN3r_SN3r = 3
        #For Azo with IL
        self.SN4a_SC1 = 4
        self.SC1_SC1 = 5
        self.SC1_TN3r = 6
        self.TN3r_TC5 = 7
        self.TC5_TC5 = 8
        self.SC1_TN2q = 9
        # for IL
        self.SX4e_SQ4n = 10
        self.SQ4n_SQ4n = 11
        self.TN2q_TN2q = 12
        self.TN2q_TC3 = 13
        self.TN2q_TC6 = 14

        self.generic_angle = 0
        self.backbone_angle_id = 1
        self.PEO_angle = 2
        self.azo_ring_angle = 3
        self.azo_cis_isomer_angle = 4
        self.azo_trans_isomer_angle = 5
        self.TFSI1_angle_typeid = 6
        self.TFSI2_angle_typeid = 7
        self.EMIM_angle_typeid = 8

        self.PEO_dihedral_typeid = 0
        self.TFSI_dihedral_typeid = 1

        self.L = L
        self.length_backbone = length_backbone
        self.num_backbone = num_backbone
        self.crosslinking_probability = crosslinking_probability
        self.length_crosslinkers = length_crosslinkers
        self.length_sidechains = length_sidechains
        self.azo_probability = azo_probability
        self.num_ionic_liquid = num_ionic_liquid

        self.positions_backbone = []
        self.positions_crosslinks = []
        self.types_backbones = []
        self.types_crosslinkers = []

        self.positions_ionic_liquid = []
        self.types_ionic_liquid = []

        #72, 54, and 36 amu
        self.R_mass = 72
        self.S_mass = 54
        self.T_mass = 36
        self.masses = []

        self.bonds = []
        self.bonds_typeids = []

        self.angles = []
        self.angles_typeids = []

        self.dihedrals = []
        self.dihedrals_typeids = []

        self.azo_architecture = azo_architecture


    def create_system(self):
        self.create_polymer_system()
        self.add_ionic_liquid()

    def create_polymer_system(self):
        # create backbones
        i = 0 # global counter for bonds
        for N in range(self.num_backbone):
            # random freely rotating chain, pretty stiff
            q = self.random_FRC(self.length_backbone+2,0.2,1.0)
            # shift center of mass of FRC for backbone to random location in box
            com = np.random.uniform(-self.L/2.0,self.L/2.0,3)
            # toss away the first two points, since they are always 0,0,0 and 1,0,0
            pos = com + q[2:]
            for j,pi in enumerate(pos):
                p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                self.positions_backbone.append(p_wrap)
                # all backbones are available for crosslinking and forming sidechains:
                self.types_backbones.append(9999) #9999 indicates availability
                self.masses.append(self.T_mass)#27.044)
                if j<len(pos)-1:
                    self.bonds.append([i,i+1])
                    self.bonds_typeids.append(self.TC1_TC1)
                if j<len(pos)-2:
                    self.angles.append([i,i+1,i+2])
                    self.angles_typeids.append(self.backbone_angle_id)
                i = i + 1

        self.num_tethered_azo = 0
        # sidechains and crosslinks
        for k,p in enumerate(self.positions_backbone):
            # only the ones with id=-1 are open
            if self.types_backbones[k] == 9999:
                # make a crosslink, PEG diacrylate (not azobenzene for now)
                random_number = np.random.uniform(0,1)
                if random_number < self.crosslinking_probability:
                    self.types_backbones[k] = self.TC1
                    start_pos = p
                    # selection function, returns id of another backbone monomer with no
                    # crosslinker on it
                    end_pos_id = self.select_crosslinker(k,self.types_backbones,self.positions_backbone)
                    self.types_backbones[end_pos_id] = self.TC1

                    # draw a gaussian bridge from start to end (random_from_to)
                    # random_from_to doesn't do PBC, so we "unwrap"
                    end_pos = self.positions_backbone[end_pos_id]
                    w = self.wrap_pbc(end_pos-start_pos,np.array([self.L,self.L,self.L]))
                    pos = self.random_from_to(1,self.length_crosslinkers+4,start_pos,start_pos+w)
                    pos = np.asarray(pos[0][1:-1])

                    # now add resulting positions, types, bonds to arrays
                    for j,pi in enumerate(pos):
                        p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                        self.positions_crosslinks.append(p_wrap)
                        if j == 0:
                            self.types_crosslinkers.append(self.SN4a)
                            self.masses.append(self.S_mass)
                            self.bonds.append([k,i])
                            self.bonds_typeids.append(self.TC1_SN4a)
                            self.bonds.append([i,i+1])
                            self.bonds_typeids.append(self.SN4a_SN3r)
                        if j != 0 and j < len(pos)-2:
                            self.types_crosslinkers.append(self.SN3r)
                            self.masses.append(self.S_mass)
                            self.bonds.append([i,i+1])
                            self.bonds_typeids.append(self.SN3r_SN3r)
                        if j == len(pos)-2:
                            self.types_crosslinkers.append(self.SN3r)
                            self.masses.append(self.S_mass)
                            self.bonds.append([i,i+1])
                            self.bonds_typeids.append(self.SN4a_SN3r)
                        if j == len(pos)-1:
                            self.types_crosslinkers.append(self.SN4a)
                            self.masses.append(self.S_mass)
                            self.bonds.append([i,end_pos_id])
                            self.bonds_typeids.append(self.TC1_SN4a)
                        #add angle associated with the new particle
                        if j == 0:
                            self.angles.append([k,i,i+1])
                            self.angles_typeids.append(self.generic_angle)
                        if j == 1:
                            self.angles.append([i-1,i,i+1])
                            self.angles_typeids.append(self.generic_angle)
                        if 1 < j and j < len(pos)-2:
                            self.angles.append([i-1,i,i+1])
                            self.angles_typeids.append(self.PEO_angle)
                        if j == len(pos) - 2:
                            self.angles.append([i-1,i,i+1])
                            self.angles_typeids.append(self.generic_angle)
                        if j == len(pos) - 1:
                            self.angles.append([i-1,i,end_pos_id])
                            self.angles_typeids.append(self.generic_angle)
                        #add dihedral associated with the new particle
                        if 0 < j and j < len(pos) - 4:
                            self.dihedrals.append([i,i+1,i+2,i+3])
                            self.dihedrals_typeids.append(self.PEO_dihedral_typeid)

                        i = i + 1
                    # print(pos)
                    # print(len(pos))
                    # print(self.types_crosslinkers)
                    # print(len(self.types_crosslinkers))
                    # print(len(self.bonds))
                    # print(len(self.bonds_typeids))
                    # break

                # add sidechain (could be PEG or Azo)
                else:
                    random_number = np.random.uniform(0,1)
                    # add EMIM-Tethered Azo sidechain
                    if random_number < self.azo_probability:
                        # increment number of IL-tethered azos
                        self.num_tethered_azo += 1

                        self.types_backbones[k] = 1
                        start_pos = p

                        length_azo = 17
                        q = self.random_FRC(length_azo+2,0.2,1.0)
                        pos = (start_pos + q[2:] - q[1])
                        # q = self.random_FRC(length_azo,0.2,1.0)
                        # pos = (start_pos + q)

                        for j,pi in enumerate(pos):
                            p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                            self.positions_crosslinks.append(p_wrap)
                            # by far the least elegant way of doing this. but it works.
                            #  0      1     2       3        4    5      6     7      8    9
                            #['SN4a','TC1','SN3r','SN3r1','SC1','TN3r','TC5','TN2q','TC6','10']
                            if j == 0:
                                self.types_crosslinkers.append(self.SN4a)
                                self.masses.append(self.S_mass)
                                self.bonds.append([k,i])
                                self.bonds_typeids.append(self.TC1_SN4a)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SN4a_SC1)
                                self.angles.append([k,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 1:
                                self.types_crosslinkers.append(self.SC1)
                                self.masses.append(self.S_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SC1_SC1)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 2:
                                self.types_crosslinkers.append(self.SC1)
                                self.masses.append(self.S_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SC1_TN3r)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 3:
                                self.types_crosslinkers.append(self.TN3r)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN3r_TC5)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 4:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TC5_TC5)
                                self.angles.append([i-1,i,i+2])
                                self.angles_typeids.append(self.azo_ring_angle)
                            if j == 5:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TC5_TC5)
                            if j == 6:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i-2])
                                self.bonds_typeids.append(self.TC5_TC5)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN3r_TC5)
                                self.angles.append([i-2,i,i+1])
                                self.angles_typeids.append(self.azo_ring_angle)
                            if j == 7:
                                self.types_crosslinkers.append(self.TN3r)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN3r_TC5)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.azo_trans_isomer_angle)
                            if j == 8:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TC5_TC5)
                                self.angles.append([i-1,i,i+2])
                                self.angles_typeids.append(self.azo_ring_angle)
                            if j == 9:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TC5_TC5)
                            if j == 10:
                                self.types_crosslinkers.append(self.TC5)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN3r_TC5)
                                self.bonds.append([i,i-2])
                                self.bonds_typeids.append(self.TC5_TC5)
                                self.angles.append([i-2,i,i+1])
                                self.angles_typeids.append(self.azo_ring_angle)
                            if j == 11:
                                self.types_crosslinkers.append(self.TN3r)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SC1_TN3r)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 12:
                                self.types_crosslinkers.append(self.SC1)
                                self.masses.append(self.S_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SC1_SC1)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 13:
                                self.types_crosslinkers.append(self.SC1)
                                self.masses.append(self.S_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SC1_TN2q)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 14:
                                self.types_crosslinkers.append(self.TN2q)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN2q_TN2q)
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.EMIM_angle_typeid)
                                self.angles.append([i-1,i,i+2])
                                self.angles_typeids.append(self.EMIM_angle_typeid)
                            if j == 15:
                                self.types_crosslinkers.append(self.TN2q)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.TN2q_TC6)
                            if j == 16:
                                self.types_crosslinkers.append(self.TC6)
                                self.masses.append(self.T_mass)
                                self.bonds.append([i,i-2])
                                self.bonds_typeids.append(self.TN2q_TC6)
                            i = i+1
                        # print(pos)
                        # print(len(pos))
                        # print(self.types_crosslinkers)
                        # print(len(self.types_crosslinkers))
                        # print(len(self.bonds))
                        # break

                    # add PEG acrylate sidechain
                    else:
                        self.types_backbones[k] = self.TC1
                        start_pos = p
                        # w = self.get_gaussian_vector(self.length_sidechains)
                        # pos = self.random_from_to(1,self.length_sidechains+2,start_pos,start_pos-w)
                        # pos = np.asarray(pos[0][1:-1])

                        q = self.random_FRC(self.length_sidechains+1+2,0.2,1.0)

                        pos = (start_pos + q[2:] - q[1])

                        for j,pi in enumerate(pos):
                            p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                            self.positions_crosslinks.append(p_wrap)
                            if j == 0:
                                self.types_crosslinkers.append(self.SN4a)
                                self.masses.append(self.S_mass)
                                self.bonds.append([k,i])
                                self.bonds_typeids.append(self.TC1_SN4a)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SN4a_SN3r)
                            if j != 0 and j < len(pos)-1:
                                self.types_crosslinkers.append(self.SN3r)
                                self.masses.append(self.S_mass)
                                self.bonds.append([i,i+1])
                                self.bonds_typeids.append(self.SN3r_SN3r)
                            if j==len(pos)-1:
                                self.types_crosslinkers.append(self.SN3r)
                                self.masses.append(self.S_mass)

                            #add angles
                            if j == 0:
                                self.angles.append([k,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if j == 1:
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.generic_angle)
                            if 1 < j and j < len(pos) - 1:
                                self.angles.append([i-1,i,i+1])
                                self.angles_typeids.append(self.PEO_angle)
                            #add dihedrals
                            if 0 < j and j < len(pos) - 3:
                                self.dihedrals.append([i,i+1,i+2,i+3])
                                self.dihedrals_typeids.append(self.PEO_dihedral_typeid)

                            i = i + 1
                        # print(self.positions_crosslinks)
                        # print(len(pos))
                        # print(self.types_crosslinkers)
                        # print(len(self.types_crosslinkers))
                        # print(len(self.bonds))
                        # break


        self.positions_crosslinks = np.asarray(self.positions_crosslinks)
        self.positions_backbone = np.asarray(self.positions_backbone)

        #self.bonds = np.asarray(self.bonds)
        self.types_backbones = np.asarray(self.types_backbones)
        self.types_crosslinkers = np.asarray(self.types_crosslinkers)

    def add_ionic_liquid(self):
        if self.azo_architecture == 'sideChainIL':
            N_to_add_EMIM = 0
            N_to_add_TFSI = self.num_tethered_azo
        else:
            print("Unknown azo_architecture")
            # existing tethered EMIM
            N_current_charge = int(len(self.types_crosslinkers[self.types_crosslinkers==self.TC6])/2.0)
            # N_current_charge = self.num_tethered_azo
            N_to_add_EMIM = int(self.num_ionic_liquid/2.0) - N_current_charge
            #TFSI
            N_to_add_TFSI = int(self.num_ionic_liquid/2.0)

        i = len(self.types_backbones) + len(self.types_crosslinkers)

        for n in range(N_to_add_TFSI):

            self.types_ionic_liquid.append(self.SX4e)
            self.types_ionic_liquid.append(self.SQ4n)
            self.types_ionic_liquid.append(self.SQ4n)
            self.types_ionic_liquid.append(self.SX4e)
            self.masses.append(self.S_mass)
            self.masses.append(self.S_mass)
            self.masses.append(self.S_mass)
            self.masses.append(self.S_mass)

            self.bonds.append([i,i+1])
            self.bonds_typeids.append(self.SX4e_SQ4n)
            self.bonds.append([i+1,i+2])
            self.bonds_typeids.append(self.SQ4n_SQ4n)
            self.bonds.append([i+2,i+3])
            self.bonds_typeids.append(self.SX4e_SQ4n)

            self.angles.append([i,i+1,i+2])
            self.angles_typeids.append(self.TFSI1_angle_typeid)
            self.angles.append([i+1,i+2,i+3])
            self.angles_typeids.append(self.TFSI2_angle_typeid)

            self.dihedrals.append([i,i+1,i+2,i+3])
            self.dihedrals_typeids.append(self.TFSI_dihedral_typeid)

            q = self.random_FRC(4,0.2,1.0)
            com = np.random.uniform(-self.L/2.0,self.L/2.0,3)
            pos = com + q

            for j,pi in enumerate(pos):
                p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                self.positions_ionic_liquid.append(p_wrap)

            i = i + 4


        for n in range(N_to_add_EMIM):

            self.types_ionic_liquid.append(self.TN2q)
            self.types_ionic_liquid.append(self.TN2q)
            self.types_ionic_liquid.append(self.TC3)
            self.types_ionic_liquid.append(self.TC6)
            self.masses.append(self.T_mass)
            self.masses.append(self.T_mass)
            self.masses.append(self.T_mass)
            self.masses.append(self.T_mass)

            self.bonds.append([i,i+1])
            self.bonds_typeids.append(self.TN2q_TN2q)
            self.bonds.append([i+1,i+2])
            self.bonds_typeids.append(self.TN2q_TC3)
            self.bonds.append([i,i+3])
            self.bonds_typeids.append(self.TN2q_TC6)
            self.bonds.append([i+1,i+3])
            self.bonds_typeids.append(self.TN2q_TC6)

            self.angles.append([i,i+1,i+2])
            self.angles_typeids.append(self.EMIM_angle_typeid)
            self.angles.append([i+3,i+1,i+2])
            self.angles_typeids.append(self.EMIM_angle_typeid)

            q = self.random_FRC(4,0.2,1.0)
            com = np.random.uniform(-self.L/2.0,self.L/2.0,3)
            pos = com + q

            for j,pi in enumerate(pos):
                p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                self.positions_ionic_liquid.append(p_wrap)

            i = i + 4
        
        if len(self.positions_ionic_liquid) == 0:
            print('No IL added')
            self.positions_ionic_liquid = np.empty(shape=(0,3))

    def save_system(self,name_gsd):
        self.output = gsd.hoomd.open(name=name_gsd, mode='w')
        frame = gsd.hoomd.Frame()

        self.positions = np.vstack((self.positions_backbone,
                                             self.positions_crosslinks,
                                             self.positions_ionic_liquid
                                            ))
        self.types = np.hstack((self.types_backbones,
                                         self.types_crosslinkers,
                                         self.types_ionic_liquid
                                        ))

        frame.particles.N = len(self.positions)

        self.positions = self.wrap_pbc(self.positions,\
                                       np.array([self.L,self.L,self.L]))
        self.scale_system(scaling=1/3)

        frame.particles.position = self.positions
                                #  0      1     2       3        4    5      6     7      8    9     10
        frame.particles.types = self.particle_types#['SN4a','TC1','SN3r','SN3r1','SC1','TN3r','TC5','TN2q','TC6','SX4e','SQ4n']
        frame.particles.typeid = self.types
        frame.particles.mass = self.masses

        frame.bonds.N = len(self.bonds)
        frame.bonds.types = self.bond_types
        frame.bonds.typeid = self.bonds_typeids
        frame.bonds.group = self.bonds

        frame.angles.N = len(self.angles)
        frame.angles.types = self.angle_types
        frame.angles.typeid = self.angles_typeids
        frame.angles.group = self.angles

        frame.dihedrals.N = len(self.dihedrals)
        frame.dihedrals.types = self.dihedral_types
        frame.dihedrals.typeid = self.dihedrals_typeids
        frame.dihedrals.group = self.dihedrals

        frame.configuration.box = [self.L,self.L,self.L,0,0,0]
        self.output.append(frame)
        self.output.close()

    def wrap_pbc(self,x,box):
        delta = np.where(x > 0.5 * box, x - box, x)
        delta = np.where(delta < - 0.5 * box, box + delta, delta)
        return delta

    def generate_random_walk(self,N):
        dims = 3
        step_n = N-1
        vec = np.random.randn(dims, step_n)
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec.T
        origin = np.zeros((1,dims))
        path = np.concatenate([origin, vec]).cumsum(0)
        return path

    def random_point_on_cone(self,R,theta,prev):
        """ Returns random vector with length R and angle theta between previos one, uniformly distributed """
        #theta *=np.pi/180.
        v = prev/np.linalg.norm(prev)
        # find "mostly orthogonal" vector to prev
        a = np.zeros((3,))
        a[np.argmin(np.abs(prev))]=1
        # find orthonormal coordinate system {x_hat, y_hat, v}
        x_hat = np.cross(a,v)/np.linalg.norm(np.cross(a,v))
        y_hat = np.cross(v,x_hat)
        # draw random rotation
        phi = np.random.uniform(0.,2.*np.pi)
        # determine vector (random rotation + rotation with theta to guarantee the right angle between v,w)
        w = np.sin(theta)*np.cos(phi)*x_hat + np.sin(theta)*np.sin(phi)*y_hat + np.cos(theta)*v
        w *=R
        return w

    def random_FRC(self,m,characteristic_ratio,bond_length):
        """ Returns a freely jointed chain with defined characteristic ratio and bond lenths. Particles can overlap """
        # build random walk in 3D with correct characteristic ratio
        # Freely rotating chain model - LJ characteristic ratio
        # is 1.88 http://dx.doi.org/10.1016/j.cplett.2011.12.040
        # c = 1+<cos theta>/(1-<cos theta>)
        # <cos theta> = (c-1)/(c+1)
        theta = np.arccos((characteristic_ratio-1)/(characteristic_ratio+1))
        coords = np.zeros((m,3))
        first_vec = np.random.randn(3)
        first_vec /= np.linalg.norm(first_vec, axis=0)
        first_vec = first_vec.T
        coords[1] = first_vec
        # coords[1]=[1,0,0]
        for i in range(2,m):
            prev = coords[i-2]-coords[i-1]
            n = self.random_point_on_cone(bond_length,theta,prev)
            new = coords[i-1]+n
            coords[i]=new

        return coords

    def get_gaussian_vector(self,N):
        x = np.random.normal(0, np.sqrt(N/3.0), 3)
        return x

    def sample_path_batch(self,M, N):
        dt = 1.0 / (N -1)
        dt_sqrt = np.sqrt(dt)
        B = np.empty((M, N), dtype=np.float32)
        B[:, 0] = 0
        for n in six.moves.range(N - 2):
             t = n * dt
             xi = np.random.randn(M) * dt_sqrt
             B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
        B[:, -1] = 0
        return B

    def random_from_to(self,M,N,start,end):
        B = []
        for i in range(3):
            B.append(self.sample_path_batch(M,N))
        B = np.asarray(B)
        res = []
        v = end-start
        d = np.linalg.norm(v)
        v = np.repeat(v, N, axis=0).reshape(3,N).T
        c = np.arange(0,N)/N
        v = v* c[:, np.newaxis]
        for i in range(M):
            b = B[:,i,:]
            b = b.T*np.sqrt(N/2.) + start
            b = b + v
            res.append(b)
        # print(start)
        # print("res:", res)
        # print(end)
        # #check the distances
        # for i in range(len(res[0])-1):
        #     bond = res[0][i] - res[0][i+1]
        #     self.wrap_pbc(bond,np.array([self.L,self.L,self.L]))
        #     dist = np.sqrt(bond[0]**2 +bond[1]**2 + bond[2]**2)
        #     print('bond distances:')
        #     print(dist)

        return res

    def select_crosslinker(self,a,cr,pos):
        pos = np.asarray(pos)
        pos = self.wrap_pbc(pos,np.array([self.L,self.L,self.L]))
        cr = np.asarray(cr)
        picked = False
        max_trials = 1000
        trials = 0
        while picked == False and trials <= max_trials:
            open_crosslinks = np.asarray(np.where(cr==9999)[0])
            Box = np.array([self.L,self.L,self.L])+1e-2

            tree = KDTree(data=pos[open_crosslinks]+Box/2., leafsize=12,boxsize=Box)
            # set maximum distance for any crosslinker to be picked from  = max extension
            Rmax = self.length_crosslinkers

            all_neigh = tree.query_ball_point(x=pos[a]+Box/2.,
                                                   r=Rmax)

            open_crosslinks_filtered = open_crosslinks[all_neigh]
            x = np.linalg.norm(self.wrap_pbc(pos[a]-pos[open_crosslinks_filtered],Box),axis=1)
            # remove "small" distances to avoid a lot of loop crosslinks
            open_crosslinks_filtered = open_crosslinks_filtered[x>Rmax/3.0]
            #x = x[x>Rmax/3.0]
            if len(open_crosslinks_filtered)>0:

                # uniformly randomly picked out of the remaining ones
                b = rng.choice(open_crosslinks_filtered,size=1,shuffle=False)

                # pick from gaussian distribution instead
                #p =np.exp(-3*x**2/(float(2*self.length_crosslinkers)))
                #p[np.isinf(p)] = 1
                #if np.sum(p)>0:
                #    p = p/np.sum(p)
                #else:
                #    p = np.ones(len(x))/len(x)
                #b = np.random.choice(open_crosslinks_filtered,p=p,size=1)

                break

            trials +=1

        else:
            print("max trials reached, pick closest availaible")
            all_dists = np.linalg.norm(
                self.wrap_pbc(pos[open_crosslinks]-pos[a],np.array([self.L,self.L,self.L])),
                axis=1)
            if len(all_dists)>0:
                pick = np.argmin(all_dists)
            else:
                print("Error - no free backbone availaibe")
                exit(1)
            #print(np.min(all_dists), pick)
            #if np.min(all_dists)>
            #smallest_dist =  sorted(all_dists)[1][0]
            #l = (all_dists==smallest_dist).flatten().reshape(1,-1)[0]
            b = [open_crosslinks[pick]]

        return b[0]

    def scale_system(self,scaling):
        # print(self.positions)
        pos = np.asarray(self.positions)
        pos = np.multiply(pos,scaling)
        self.positions = pos
        self.L = self.L*scaling
        # print(self.positions)
        return
