import numpy as np
import gsd,gsd.hoomd
from collections import defaultdict
import freud 

# make the system class
class System:
    def __init__(self):
        ## define chemical tags ----
        self.particles_types = ['SN3r','SX4e','SQ4n','TC3','TN2q', 'TC6']
        ## ---------------[  0  , 1  ,     2     ,  3 ,     4     ,   5]
        self.bond_types = ['PEO','SX4e-SQ4n','SQ4n-SQ4n','TC3-TN2q','TN2q-TN2q','TN2q-TC6']
        self.angle_types = ['PEO','TFSI','EMIM']
        self.dihedral_types = ['PEO','TFSI']

        self.thiol_type = 0
        self.ene_type = 1
        self.radical_thiol = 2
        self.radical_carbon = 3
        self.sulfur = 4
        self.carbon = 5
        self.spacer_type1 = 6
        self.spacer_type2 = 7

        # overwritten by "create_initial_configuration"
        self.N_particles = 0
        self.N_dummy_bonds = 0
        self.N_dummy_angles = self.N_dummy_bonds*2
        
        self.size_monomer = 0 
        self.size_extender = 0 
        self.size_crosslinker = 0

        # overwritten by "propagate_reaction" 
        self.chain_transfer_probability = 0.0
        self.thiol_reaction_probability = 0.0
        self.chain_side_reaction_probability = 0.0
        self.rcut_neigh = 0

        self.FJ_system = False
        self.unspaced_system = True

    '''
    create the initial configuration in the box
    args: the system, density, amount of crosslinker, amount of ene monomers, monomer size, and extender size
    returns: initial gsd frame
    two types of systems
    1. Spaced:
    2. Unspaced:

    Systems can also be FJ or not, determining whether angles are present
    '''
    def create_initial_configuration(self,density,crosslinker,N_monomers, monomer_size=0, extender_size=0):
        # if the system is not the unspaced system, add extenders and change the monomer size
        if self.size_extender>0 and self.size_monomer>0:
            self.size_monomer = monomer_size+8
            self.size_extender = extender_size+1
            self.unspaced_system = False
        # otherwise use the dumbell system
        else:
            self.size_monomer = 4 
            self.size_extender = 2
            self.size_crosslinker = 5
            self.unspaced_system = True
        
        ### in this system, equal-stoichiometric ene-thiol
        # define the number of monomers, reactive beads, and the crosslinker fraction
        self.N_monomers = int(N_monomers)
        reactive_beads = self.N_monomers*2 
        crosslinker_fraction = crosslinker/100.
        self.N_crosslinker = int(np.round(reactive_beads*crosslinker_fraction/4.))
        self.N_extenders =  int(np.round((reactive_beads-self.N_crosslinker*4)/2.))

        print("N Monomers A = ",self.N_monomers)
        print("N Monomers B = ", self.N_extenders)
        print("N crosslinkers B =", self.N_crosslinker)
        print("Balance: 2*%s + 4*%s = 2*%s"%(self.N_extenders,self.N_crosslinker,self.N_monomers))

        # calculate the overall tallies in the box
        self.N_particles = self.N_crosslinker*self.size_crosslinker +\
                        self.N_extenders*self.size_extender+\
                        self.N_monomers*self.size_monomer
        self.L =  (self.N_particles/density)**(1/3.0)
        print("Box length = %1.2f"%(self.L))
        print("total particles = %d"%(self.N_particles))
        print("number density = %1.3f"%(self.N_particles/(self.L**3)))
        # self.N_dummy_bonds = int(np.round(1.1*self.N_particles))
        self.N_dummy_bonds = int(np.round(reactive_beads*1.05))
        self.N_dummy_angles = self.N_dummy_bonds*3


        all_positions = []
        all_types = []
        all_bonds = []
        all_bonds_types = []
        all_angles = []
        # all_angles_types = []
        '''
        ### crosslinkers 
        ## unspaced system
        '''
        if self.unspaced_system == False: 
            one_set_types = np.array([self.thiol_type,self.spacer_type1,self.spacer_type2,
                                self.thiol_type,self.spacer_type1,self.spacer_type2,
                                self.thiol_type,self.spacer_type1,self.spacer_type2,
                                self.thiol_type,self.spacer_type1,self.spacer_type2,
                                self.spacer_type1])
            one_set_bonds = np.array([[0,1],[1,2],
                                    [3,4],[4,5],
                                    [6,7],[7,8],
                                    [9,10],[10,11],
                                    [2,12],[5,12],[8,12],[11,12]])
            
            one_set_angles = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],
                                    [1,2,12],[4,5,12],[7,8,12],[10,11,12]])

            for i in range(self.N_crosslinker):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_crosslinker))*0.5
                position = np.repeat(p,self.size_crosslinker,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_crosslinker*i)
                for b in one_set_bonds:
                    all_bonds_types.append(0)
                for a in one_set_angles:
                    all_angles.append(a+self.size_crosslinker*i)
                # for a in one_set_angles:
                #     all_angles_types.append(0)
        ## unspaced system
        else: 
            one_set_types = np.array([self.thiol_type,
                                    self.thiol_type,
                                    self.thiol_type,
                                    self.thiol_type,
                                    self.spacer_type1])
            
            one_set_bonds = np.array([[0,4],[1,4],[2,4],[3,4]])
            
            if self.FJ_system == False:
                one_set_angles = np.array([[0,4,1],[1,4,2],[2,4,3],[3,4,0]])

            for i in range(self.N_crosslinker):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_crosslinker))*0.5
                position = np.repeat(p,self.size_crosslinker,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_crosslinker*i)
                for b in one_set_bonds:
                    all_bonds_types.append(0)

            

        
        ### thiol chain-extenders
        ## non-spaced system
        if self.unspaced_system == False: 
            one_set_types = []
            one_set_types.append(self.thiol_type)
            for i in range(self.size_extender-2):
                one_set_types.append(self.spacer_type1) #C
            one_set_types.append(self.thiol_type)

            a = np.arange(0,self.size_extender-1)
            b = np.arange(1,self.size_extender-0)
            one_set_bonds = np.vstack((a,b)).T

            if self.FJ_system == False:
                a = np.arange(0,self.size_extender-2)
                b = np.arange(1,self.size_extender-1)
                c = np.arange(2,self.size_extender-0)
                one_set_angles = np.vstack([a,b,c]).T

            N_current = self.N_crosslinker*self.size_crosslinker

            for i in range(self.N_extenders):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_extender))*0.5
                position = np.repeat(p,self.size_extender,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_extender*i+N_current)
                for b in one_set_bonds:
                    all_bonds_types.append(0)

                if self.FJ_system == False:
                    for a in one_set_angles:
                        all_angles.append(a+self.size_extender*i+N_current)
                    # for a in one_set_angles:
                    #     all_angles_types.append(0)
        ## unspaced system
        else: 
            one_set_types= [self.thiol_type,self.thiol_type]
            one_set_bonds = np.array([[0,1]])

            N_current = self.N_crosslinker*self.size_crosslinker
            
            for i in range(self.N_extenders):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_extender))*0.5
                position = np.repeat(p,self.size_extender,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_extender*i+N_current)
                for b in one_set_bonds:
                    all_bonds_types.append(0)
        '''
        ### monomers 
        ## non-spaced system
        '''
        if self.unspaced_system == False: 
            one_set_types = []
            one_set_types.append(self.ene_type)
            one_set_types.append(self.ene_type)
            one_set_types.append(self.spacer_type2) #D
            one_set_types.append(self.spacer_type2)
            for i in range(self.size_monomer-6):
                one_set_types.append(self.spacer_type1) #C
            one_set_types.append(self.spacer_type2)
            one_set_types.append(self.spacer_type2)
            one_set_types.append(self.ene_type)
            one_set_types.append(self.ene_type)
            
            a = np.arange(0,self.size_monomer-1)
            b = np.arange(1,self.size_monomer-0)
            one_set_bonds = np.vstack((a,b)).T

            if self.FJ_system == False:
                a = np.arange(0,self.size_monomer-2)
                b = np.arange(1,self.size_monomer-1)
                c = np.arange(2,self.size_monomer-0)
                one_set_angles = np.vstack([a,b,c]).T

            N_current += self.N_extenders*self.size_extender

            for i in range(self.N_monomers):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_monomer))*0.5
                position = np.repeat(p,self.size_monomer,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_monomer*i+N_current)
                for n,b in enumerate(one_set_bonds):
                    # terminal double bonds 
                    if n==0 or n==len(one_set_bonds)-1:
                        all_bonds_types.append(2)
                    # internal single bonds
                    else:
                        all_bonds_types.append(4)
                if self.FJ_system == False:
                    for a in one_set_angles:
                        all_angles.append(a+self.size_monomer*i+N_current)
                # for a in one_set_angles:
                #     all_angles_types.append(0)
        else: 
            # types of atoms in the order they appear in line
            one_set_types= [self.ene_type,self.ene_type,self.ene_type,self.ene_type]
            # which atoms are bonded together
            one_set_bonds = np.array([[0,1],[1,2],[2,3]])

            if self.FJ_system == False:
                one_set_angles = np.array([[0,1,2],[1,2,3]])

            N_current += self.N_extenders*self.size_extender
            
            # for each monomer in the system, generate the poistions, and the bonds to each other
            for i in range(self.N_monomers):
                p = np.random.uniform(-self.L/2., +self.L/2., size=(1,3))
                offsets = self.sample_spherical(int(self.size_monomer))*0.5
                position = np.repeat(p,self.size_monomer,axis=0) + offsets.T
                for r in position:
                    all_positions.append(r)
                for q in one_set_types:
                    all_types.append(q)
                for b in one_set_bonds:
                    all_bonds.append(b+self.size_monomer*i+N_current)
                for n,b in enumerate(one_set_bonds):
                    # terminal double bonds
                    if n==0 or n==len(one_set_bonds)-1:
                        all_bonds_types.append(2)
                    # internal single bonds
                    else:
                        all_bonds_types.append(4)
                if self.FJ_system == False:
                    for a in one_set_angles:
                        all_angles.append(a+self.size_monomer*i+N_current)
        
        # format all molecular information
        all_positions = np.asarray(all_positions)
        all_positions = self.wrap_pbc(all_positions)
        all_types = np.asarray(all_types)
        all_bonds = np.asarray(all_bonds)
        all_bonds_types = np.asarray(all_bonds_types)
        all_angles = np.asarray(all_angles)
        # all_angles_types = np.asarray(all_angles_types)
        
        # write all information into gsd file 
        frame = gsd.hoomd.Frame()
        frame.particles.N = self.N_particles+4

        frame.particles.position = np.zeros(shape=(self.N_particles+4,3))
        frame.particles.position[:self.N_particles] = all_positions
        frame.particles.typeid = np.hstack((all_types, np.asarray([self.dummy_type,self.dummy_type,self.dummy_type,self.dummy_type])))

        frame.particles.types = self.particles_types
        frame.bonds.types = self.bond_types
        frame.angles.types = self.angle_types


        # throw the dummy particles in a corner of the box
        eps=1e-6
        Lo = self.L/2.0-eps
        
        frame.particles.position[-1]=[Lo,Lo,Lo]
        frame.particles.position[-2]=[Lo,Lo,Lo]
        frame.particles.position[-3]=[Lo,Lo,Lo]
        frame.particles.position[-4]=[Lo,Lo,Lo]

        frame.configuration.box = [self.L, self.L, self.L, 0, 0, 0]

        bonds = np.asarray((all_bonds.flatten()).reshape(-1,2))
        dummy_bonds = np.tile([self.N_particles,self.N_particles+1],(self.N_dummy_bonds,1))
        frame.bonds.group = np.vstack((bonds,dummy_bonds))

        frame.bonds.typeid = np.hstack((all_bonds_types,len(dummy_bonds)*[self.dummy_type_bond]))
        frame.bonds.N = len(bonds)+len(dummy_bonds)
        
        # if includng angles, specify them in the gsd file
        if self.FJ_system == False: 
            angles = np.asarray((all_angles.flatten()).reshape(-1,3))
            dummy_angles = np.tile([self.N_particles,self.N_particles+1,self.N_particles+2],(self.N_dummy_angles,1))
            frame.angles.group = np.vstack((angles,dummy_angles))
            frame.angles.typeid = np.hstack((len(angles)*[0],len(dummy_angles)*[self.dummy_type_angle]))
            frame.angles.N = len(angles)+len(dummy_angles)
        return frame

    def set_types(self):
        self.types = ['A','B','X']
        self.type_list = []

        for n in range(self.num_pol):
            for m in range(self.num_mon):
                if m<self.NA:
                    self.type_list.append('A')
                elif m<self.NA+self.NB:
                    self.type_list.append('B')
                else:
                    self.type_list.append('A')

        for i in range(self.num_field_particles):
            self.type_list.append('X')

        # set typeid per particle
        map_types = {t:i for i, t in enumerate(self.types)}
        self.typeid = np.array([map_types[t] for t in self.type_list], dtype=np.int32)

    def set_bonds(self):
        join = lambda first, second: ''.join(sorted(first + second))
        # set bond types first
        self.bond_types = ['AA','AB','BB']
        # for i in range(len(self.type_list)-1):
        #     bond_type = join(self.type_list[i], self.type_list[i+1])
        #     if bond_type not in self.bond_types:
        #         self.bond_types.append(bond_type)

        # set bond typeids and groups as in hoomd
        map_bonds = {t:i for i, t in enumerate(self.bond_types)}
        self.bond_typeid = []
        self.bond_group  = []
        for i in range(self.num_pol):
            for j in range(self.num_mon-1):
                k = i*self.num_mon + j
                # connects to the next monomer along the backbone
                self.bond_typeid.append(map_bonds[join(self.type_list[k], self.type_list[k+1])])
                self.bond_group.append([k, k+1])
        self.bond_typeid = np.asarray(self.bond_typeid, dtype=np.int32)
        self.bond_group = np.asarray(self.bond_group, dtype=np.int32)

    def set_positions(self):
        self.N_current = 0
        while self.N_current < self.num_pol:
            self.add_pol()
            self.N_current += 1

    # Assumes 3D
    def add_pol(self, threshold=1.5):
        u = np.zeros((1,3))
        u[:,0] = np.random.uniform(-self.Lx/2., self.Lx/2.)
        u[:,1] = np.random.uniform(-self.Ly/2., self.Ly/2.)
        u[:,2] = np.random.uniform(-self.Lz/2., self.Lz/2.)
        r = random_FRC(self.num_mon,self.c,self.bond)
        r = r + u
        if self.N_current > 0:
            self.positions = np.vstack((self.positions, r))
        else:
            self.positions = r
    
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

    def get_snap(self, context):
        with context:
            snap = make_snapshot(N=self.N,
                                 particle_types=self.types,
                                 bond_types=self.bond_types,
      		                     box=self.box)

            self.positions = self.wrap_pbc(self.positions)
    	    # set typeids, masses and positions
            print('self.N',self.N)
            print('len(self.positions)',len(self.positions))
            for k in range(self.N):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
                snap.particles.mass[k] =  1.0 if self.typeid[k] == 0 else 1.6
                #hard coded
            # set bond typeids and groups
            snap.bonds.resize(len(self.bond_group))
            for k in range(len(self.bond_group)):
                snap.bonds.typeid[k] = self.bond_typeid[k]
                snap.bonds.group[k] = self.bond_group[k]

        return snap

    def sample_spherical(self,npoints, ndim=3):
        """
        Draw npoints random numbers on a sphere in ndim. 

        """
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
    

    def connected_components(self,lists):
        """
        merges lists with common elements

        args: bonds from configuration (frame.bonds.group)

        returns: list of connected particles by id

        Useful for finding bonded particles in a configuration, tested for
        linear polymers with consecutive bonds (0-1-2-3-4-5, 6-7-8-9-10,..)
        and non consecutive ids ( 0-5-6-8-10, 1-4-3-2-9,...) but no other
        configuration yet. Works with ints as well as str.

        """
        neighbors = defaultdict(set)
        seen = set()
        for each in lists:
            for item in each:
                neighbors[item].update(each)
        def component(node, neighbors=neighbors, seen=seen, see=seen.add):
            nodes = set([node])
            next_node = nodes.pop
            while nodes:
                node = next_node()
                see(node)
                nodes |= neighbors[node] - seen
                yield node
        for node in neighbors:
            if node not in seen:
                yield sorted(component(node))


    def wrap_pbc(self,delta):
        Box = np.array([self.L,self.L,self.L])
        delta = np.where(delta > 0.5 * Box, delta-Box, np.where(delta < -0.5 * Box, delta+Box , delta))
        return delta
    
