#Modified for compatibility with Hoomd 4.6.0

import warnings
import numpy as np
import os,sys
import gsd.hoomd
import hoomd
import sys, math
import csv
import martini3
from martini3 import molecules

class RandomPolymerMaker():

    def __init__(self,Ntotal,nPEO,nAzo,box,seq,id,name_a,name_b):#b,
        self.nPEO = nPEO
        self.nAzo = nAzo
        self.num_mon=nPEO+7*nAzo+2
        self.num_beads = nPEO*
        self.num_pol = Ntotal
        self.N = self.num_pol*self.num_mon
        self.bond = 1
        self.box = [box[0],box[1],box[2],0,0,0]
        self.Lx = box[0]
        self.Ly = box[1]
        self.Lz = box[2]
        self.particle_types = ['PEO','Benz','NN','Endgroup']
        self.peo_id = 0
        self.benz_id = 1
        self.nn_id = 2
        self.endgroup_id = 3

        peoInterval = self.nPEO/(self.nAzo+1)
        self.azoIndices = []
        for i in range(self.nAzo):
            self.azoIndices.append(round(peoInterval*(i+1)))

        self.positions  = []
        # set types, bonds, positions
        self.set_types()
        self.set_bonds()
        self.set_angles()
        self.set_positions()


    # def set_types(self):
    #     self.type_list = []

    #     # for n in range(self.num_pol):
    #     #     for m in range(self.num_mon):
    #     #         if m < self.nPEO/2:
    #     #             self.type_list.append('PEO')
    #     #         elif m < self.nPEO/2+1:
    #     #             self.type_list.append('Benz')
    #     #         elif m < self.nPEO/2+2:
    #     #             self.type_list.append('NN')
    #     #         elif m < self.nPEO/2+1:
    #     #             self.type_list.append('Benz')
    #     #         else:
    #     #             self.type_list.append('PEO')
    #     for n in range(self.num_pol):
    #         self.type_list.append('Endgroup')
    #         for m in range(self.nPEO):
    #             if m in self.azoIndices:
    #                 self.type_list.append('Benz')
    #                 self.type_list.append('Benz')
    #                 self.type_list.append('Benz')
    #                 self.type_list.append('NN')
    #                 self.type_list.append('Benz')
    #                 self.type_list.append('Benz')
    #                 self.type_list.append('Benz')
    #             self.type_list.append('PEO')
    #         self.type_list.append('Endgroup')

    #     # set typeid per particle
    #     map_types = {t:i for i, t in enumerate(self.particle_types)}
    #     self.typeid = np.array([map_types[t] for t in self.type_list], dtype=np.int32)

    # def set_bonds(self):
    #     # join = lambda first, second: ''.join(sorted(first + second))
    #     join = lambda first, second: ''.join(sorted([first, second]))
    #     # set bond types first
    #     self.bond_types = []
    #     for i in range(len(self.type_list)-1):
    #         bond_type = join(self.type_list[i], self.type_list[i+1])
    #         if bond_type not in self.bond_types:
    #             self.bond_types.append(bond_type)
    #     '''
    #     self.bond_types.append('Xlink')
    #     self.bond_types.append('Dummy')
    #     '''
    #     print("bond_types initialized:",self.bond_types)

    #     # set bond typeids and groups as in hoomd
    #     map_bonds = {t:i for i, t in enumerate(self.bond_types)}
    #     self.bond_typeid = []
    #     self.bond_group  = []
    #     for i in range(self.num_pol):
    #         for j in range(self.num_mon-1):
    #             k = i*self.num_mon + j
    #             # connects to the next monomer along the backbone
    #             self.bond_typeid.append(map_bonds[join(self.type_list[k], self.type_list[k+1])])
    #             self.bond_group.append([k, k+1])

    #             #checks if a ring needs to be made
    #             if (self.typeid[k-1] == self.benz_id) and (self.typeid[k] == self.benz_id) and (self.typeid[k+1] == self.benz_id):
    #                 self.bond_typeid.append(map_bonds[join(self.type_list[k-1], self.type_list[k+1])])
    #                 self.bond_group.append([k-1, k+1])
    #     '''
    #     # for i in range(self.num_pol):
    #     #     for j in range(self.num_mon-1):
    #     #         k = i*self.num_mon + j
    #     #         # connects to the next monomer along the backbone
    #     #         self.bond_typeid.append(map_bonds[join(self.type_list[k], self.type_list[k+1])])
    #     #         self.bond_group.append([k, k+1])
    #     '''
    #     '''
    #     #Add dummy bonds for use in crosslinking
    #     for i in range(200):
    #         self.bond_typeid.append(map_bonds['Dummy'])
    #         self.bond_group.append([0,1])
    #     '''

    #     self.bond_typeid = np.asarray(self.bond_typeid, dtype=np.int32)
    #     self.bond_group = np.asarray(self.bond_group, dtype=np.int32)

    # def set_angles(self):
    #     self.angle_types = ['cisAzo','transAzo','polymer','benzPara','benzInt']
    #     map_angles = {t:i for i, t in enumerate(self.angle_types)}
    #     self.angle_typeid = []
    #     self.angle_group  = []
    #     # NNids = range(0,self.N)[np.where(self.type_list == 'NN')]
    #     # print(NNids)
    #     for i in range(self.num_pol):
    #         for j in range(1,self.num_mon-1):
    #             k = i*self.num_mon + j
    #             if self.type_list[k] == 'Benz':
    #                 if self.type_list[k-1] == 'PEO' and self.type_list[k+2] == 'Benz':
    #                     #First benzene particle
    #                     self.angle_typeid.append(map_angles['benzPara'])
    #                     self.angle_group.append([k-1,k,k+2])
    #                 elif self.type_list[k-2] == 'Benz' and self.type_list[k+1] == 'NN':
    #                     #Third benzene particle
    #                     self.angle_typeid.append(map_angles['benzPara'])
    #                     self.angle_group.append([k-2,k,k+1])
    #                 elif self.type_list[k-2] == 'Benz' and self.type_list[k+1] == 'NN':
    #                     #Middle benzene particle
    #                     self.angle_typeid.append(map_angles['benzInt'])
    #                     self.angle_group.append([k-1,k,k+1])
    #                 else:
    #                     warnings.warn("Entered unknown benzene bead angle condition")
    #             elif self.type_list[k] == 'NN':
    #                 # self.angle_typeid.append('transAzo')
    #                 self.angle_typeid.append(map_angles['transAzo'])
    #                 self.angle_group.append([k-1,k,k+1])
    #             else:
    #                 # self.angle_typeid.append('polymer')
    #                 self.angle_typeid.append(map_angles['polymer'])
    #                 self.angle_group.append([k-1,k,k+1])
    
    #     self.angle_typeid = np.asarray(self.angle_typeid, dtype=np.int32)
    #     self.angle_group = np.asarray(self.angle_group, dtype=np.int32)

    # def set_positions(self):
    #     self.N_current = 0
    #     while self.N_current < self.num_pol:
    #         self.add_pol()
    #         self.N_current += 1

    # def add_pol(self, threshold=1.5):
    #     r = self.generate_random_walk(self.num_mon)
    #     u = np.zeros((1,3))
    #     u[:,0] = np.random.uniform(-self.Lx/2., self.Lx/2.)
    #     u[:,1] = np.random.uniform(-self.Ly/2., self.Ly/2.)
    #     u[:,2] = np.random.uniform(-self.Lz/2., self.Lz/2.)
    #     r = r + u
    #     if self.N_current > 0:
    #         self.positions = np.vstack((self.positions, r))
    #     else:
    #         self.positions = r
    #         #check pbc

    def wrap_pbc(self,x):
        box = np.array([self.Lx,self.Ly,self.Lz])
        delta = np.where(x > 0.5 * box, x - box, x)
        delta = np.where(delta < - 0.5 * box, box + delta, delta)
        return delta

    # Assumes 3D
    def add_pol(self,seq,id,name_a,name_b,threshold=1.5):
        for i in range(self.num_pol):
            #Generate the random position and orientation of the polymer to be added
            rand_pos = np.zeros((self.num_pol,3))
            rand_pos[:,0] = np.random.uniform(-self.Lx/2., self.Lx/2.)
            rand_pos[:,1] = np.random.uniform(-self.Ly/2., self.Ly/2.)
            rand_pos[:,2] = np.random.uniform(-self.Lz/2., self.Lz/2.)
            # rand_orientation = np.random.uniform(low=-1,high=1,size=3)
            is_inverted = np.random.rand() > 0.5

            #initialize the molecule positions
            contents = molecules.Contents()
            molecule = molecules.make_seq_def(
                    contents,
                    name_a,
                    name_b,
                    seq,
                    id,
                    x_shift=rand_pos[0],
                    y_shift=rand_pos[1],
                    z_shift=rand_pos[2],
                    is_inverted=is_inverted,
                    bond_angle=135
                )
            molecule_positions = np.array(molecule.position)
            self.wrap_pbc(molecule_positions)

            


    def add_azobenzene(self,origin,orientation):
        """
        Read a csv with information about beads present in a molecule and
        translate it into information python can read

        Args:
        molecule_beads (string): string detailing csv with molecule information.
        current_index (list of Int): current_index[0] = bead index. current_index[1] = bond index. current_index[2] = angle_index


        Returns:
        idx (list of ints): bead index
        bead_types (list of strings): bead types (i.e. 'Q1', 'W')
        positions (tuple of (x,y,z)): tuple containing relative x,y,z positions for the initialization of a molecule

        """
        idx = []
        bead_types = []
        positions = []
        charges = []
        with open('azobenzene.csv', newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "Index":
                    idx.append(str(int(row[0]) + current_index[0]))
                    bead_types.append(row[1])
                    positions.append([float(row[2]), float(row[3]), float(row[4])])
                    if "Q" in row[1]:
                        charges.append(row[5])
                    else:
                        charges.append(None)
        return idx, bead_types, positions, charges

    def generate_random_walk(self,N):
        dims = 3
        step_n = N-1
        vec = np.random.randn(dims, step_n)
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec*self.bond
        vec = vec.T
        origin = np.zeros((1,dims))
        path = np.concatenate([origin, vec]).cumsum(0)
        return path

    def get_frame(self):
        frame = gsd.hoomd.Frame()

        frame.configuration.box = self.box

        frame.particles.N = self.N
        frame.particles.types = self.particle_types
        self.positions = self.wrap_pbc(self.positions)
        frame.particles.position = self.positions
        frame.particles.typeid = self.typeid
        frame.particles.mass = [0]*self.N
        # set masses based on the particle typeid
        for k in range(self.N):
            frame.particles.mass[k] =  1.0
            #hard coded TODO
        
        # set bond typeids and groups
        frame.bonds.N = len(self.bond_typeid)                    #THIS IS NEW
        frame.bonds.types = self.bond_types
        frame.bonds.typeid = self.bond_typeid
        frame.bonds.group = self.bond_group

        frame.angles.N = len(self.angle_typeid)
        frame.angles.types = self.angle_types
        frame.angles.typeid= self.angle_typeid
        frame.angles.group = self.angle_group
        return frame

    # def get_snap(self, context):
    #     #Not completely sure, but it seems like Hoomd3 is without the typical context manager
    #     # the subclass is gone, but migration guide suggests the functionality is intact, just implicit
    #     # with context:
    #         # snap = make_snapshot(N=self.N,
    #         #                     particle_types=self.particle_types,
    #         #                     bond_types=self.bond_types,
    #   	    #                     box=self.box)
    #         snap = gsd.hoomd.Snapshot()
    #         snap.particles.N = self.N
    #         snap.particles.types = self.particle_types
    #         snap.bonds.N = (self.num_mon-1)*self.num_pol                    #THIS IS NEW
    #         snap.bonds.types = self.bond_types
    #         snap.configuration.box = self.box

    #         self.positions = self.wrap_pbc(self.positions)
    #         snap.particles.position = self.positions
    #         snap.particles.typeid = self.typeid
    #         snap.particles.mass = [0]*self.N

    #         # set typeids, masses and positions
    #         for k in range(self.N):
    #             snap.particles.mass[k] =  1.0
    #             #hard coded TODO
    #         # set bond typeids and groups
    #         snap.bonds.typeid = self.bond_typeid
    #         snap.bonds.group = self.bond_group

    #         return snap
