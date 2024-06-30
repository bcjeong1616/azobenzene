#I THINK WE ALSO NEED A NEW PARTICLE TYPE, CORRESPONDING TO THE REACTED ENE, BUT NOT A RADICAL

import numpy as np
import gsd,gsd.hoomd
from collections import defaultdict
import freud 

class System:
    def __init__(self):

        self.particles_types = ['Thiol','Ene','RSulfur','RCarbon','C','D','Dummy']
        self.bond_types = ['A','B','C','D','E','Dummy']
        self.angle_types = ['A','B','C','Dummy']

        # needs to be the correct integer position of 'dummy' in the arrays above 
        self.dummy_type = len(self.particles_types)-1
        self.dummy_type_bond = len(self.bond_types)-1
        self.dummy_type_angle = len(self.angle_types)-1

        self.thiol_type = 0
        self.ene_type = 1
        self.radical_thiol = 2 
        self.radical_carbon = 3
        self.spacer_type1 = 4
        self.spacer_type2 = 5

        
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

        self.simple_system = True 

    def create_initial_configuration(self,density,crosslinker,N_monomers,
                                     monomer_size=0, extender_size=0):
    
        if self.size_extender>0 and self.size_monomer>0:
            self.size_monomer = monomer_size+6 
            self.size_extender = extender_size+1 
            self.simple_system = False 
        else:
            self.size_monomer = 2 
            self.size_extender = 2
            self.size_crosslinker = 5
            self.simple_system = True

        self.N_monomers = int(N_monomers)
        reactive_beads = self.N_monomers*2 
        crosslinker_fraction = crosslinker/100.

        self.N_crosslinker = int(np.round(reactive_beads*crosslinker_fraction/4.))
        self.N_extenders =  int(np.round((reactive_beads-self.N_crosslinker*4)/2.))

        print("N Monomers A = ",self.N_monomers)
        print("N Monomers B = ", self.N_extenders)
        print("N crosslinkers B =", self.N_crosslinker)
        print("Balance: 2*%s + 4*%s = 2*%s"%(self.N_extenders,self.N_crosslinker,self.N_monomers))
        
        self.N_particles = self.N_crosslinker*self.size_crosslinker +\
                           self.N_extenders*self.size_extender+\
                           self.N_monomers*self.size_monomer
        
        self.L =  (self.N_particles/density)**(1/3.0)
        print("Box length = %1.2f"%(self.L))
        print("total particles = %d"%(self.N_particles))
        print("number density = %1.3f"%(self.N_particles/(self.L**3)))
        self.N_dummy_bonds = int(np.round(1.1*self.N_particles))
        self.N_dummy_angles = self.N_dummy_bonds*2

        all_positions = []
        all_types = []
        all_bonds = []
        all_bonds_types = []
        all_angles = []

        # crosslinkers 
        if self.simple_system == False: 
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
        else: 
            one_set_types = np.array([self.thiol_type,
                                    self.thiol_type,
                                    self.thiol_type,
                                    self.thiol_type,
                                    self.spacer_type1])
            
            one_set_bonds = np.array([[0,4],[1,4],[2,4],[3,4]])
            
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

        # extenders     
        if self.simple_system == False: 
            one_set_types = []
            one_set_types.append(self.thiol_type)
            for i in range(self.size_extender-2):
                one_set_types.append(self.spacer_type1) #C
            one_set_types.append(self.thiol_type)

            a = np.arange(0,self.size_extender-1)
            b = np.arange(1,self.size_extender-0)
            one_set_bonds = np.vstack((a,b)).T

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
                for a in one_set_angles:
                    all_angles.append(a+self.size_extender*i+N_current)
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
                    
        
        # monomers     
        if self.simple_system == False: 
            one_set_types = []
            one_set_types.append(self.ene_type)
            one_set_types.append(self.spacer_type2) #D
            one_set_types.append(self.spacer_type2)
            for i in range(self.size_monomer-6):
                one_set_types.append(self.spacer_type1) #C
            one_set_types.append(self.spacer_type2)
            one_set_types.append(self.spacer_type2)
            one_set_types.append(self.ene_type)
            
            a = np.arange(0,self.size_monomer-1)
            b = np.arange(1,self.size_monomer-0)
            one_set_bonds = np.vstack((a,b)).T

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
                for b in one_set_bonds:
                    all_bonds_types.append(2)
                for a in one_set_angles:
                    all_angles.append(a+self.size_monomer*i+N_current)
        else: 
            one_set_types= [self.ene_type,self.ene_type]
            one_set_bonds = np.array([[0,1]])

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
                for b in one_set_bonds:
                    all_bonds_types.append(2)

        all_positions = np.asarray(all_positions)
        all_positions = self.wrap_pbc(all_positions)
        all_types = np.asarray(all_types)
        all_bonds = np.asarray(all_bonds)
        all_bonds_types = np.asarray(all_bonds_types)
        all_angles = np.asarray(all_angles)
        
        frame = gsd.hoomd.Frame()
        frame.particles.N = self.N_particles+4

        frame.particles.position = np.zeros(shape=(self.N_particles+4,3))
        frame.particles.position[:self.N_particles] = all_positions
        frame.particles.typeid = np.hstack((all_types, np.asarray([self.dummy_type,self.dummy_type,self.dummy_type,self.dummy_type])))

        frame.particles.types = self.particles_types
        frame.bonds.types = self.bond_types
        frame.angles.types = self.angle_types

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
        
        if self.simple_system == False: 
            angles = np.asarray((all_angles.flatten()).reshape(-1,3))
            dummy_angles = np.tile([self.N_particles,self.N_particles+1,self.N_particles+2],(self.N_dummy_angles,1))
            frame.angles.group = np.vstack((angles,dummy_angles))
            frame.angles.typeid = np.hstack((len(angles)*[0],len(dummy_angles)*[self.dummy_type_angle]))
            frame.angles.N = len(angles)+len(dummy_angles)

        return frame 
    

    def flip_radicals_on(self,snapshot,radical_number_percent):

        # TODO: do we need to ensure only one thiol per monomer is selected or can both ends be reactive?
        
        # turn some A into radical_thiol
        ids = np.arange(snapshot.particles.N) 
        all_thiols =  ids[snapshot.particles.typeid==self.thiol_type]
        n_radicals = int(np.round(radical_number_percent*len(all_thiols)/100.0))
        radicals = np.random.choice(all_thiols, n_radicals, replace=False)

        snapshot.particles.typeid[radicals] = self.radical_thiol


    def bond_formation(self,snapshot,box,positions,particle_ids,ids,idx):

        if self.thiol_reaction_probability>0:
            # Propagation/bond formation step 
            radicals_positions = positions[particle_ids==self.radical_thiol]
            radicals_ids = ids[particle_ids==self.radical_thiol]
            polymer_ids = ids[particle_ids==self.ene_type]  # all Enes

            # bond type = 2 unreacted ene = carbon double bond. 
            # bond type = 4 reacted ene =  carbon single bond - can't react in bond formation
            # identify polymer candidates as ene particles with only one bond on them  
            # I THINK THIS IS THE BUG, IT SHOULD INCLUDE BOND TYPE 4
            bond_ids,bond_counts = np.unique((snapshot.bonds.group[(snapshot.bonds.typeid==2) | (snapshot.bonds.typeid==4)]).flatten(),return_counts=True)
            only_one_bond_ids = bond_ids[bond_counts==1]
            polymer_candidates_ids = np.intersect1d(only_one_bond_ids,polymer_ids)
            polymer_candidates_positions = positions[polymer_candidates_ids]

            if len(radicals_ids)>0 and len(polymer_candidates_ids)>0:
                # find all pairs of radicals and reactable enes within rcut_neigh distance between them
                aq = freud.locality.AABBQuery(box, radicals_positions)
                nlist = aq.query(polymer_candidates_positions, {'r_max': self.rcut_neigh, 'exclude_ii':True}).toNeighborList()
                all_pairs = nlist[:]

                # get the tags of reactive particles that are within the cutoff
                actual_ids_polymers = polymer_candidates_ids[all_pairs[:,0]]
                actual_ids_radicals = radicals_ids[all_pairs[:,1]]
                actual_ids = np.vstack((actual_ids_radicals,actual_ids_polymers)).T

                for a in np.unique(actual_ids_radicals):

                    neigh_a = actual_ids[actual_ids[:,0]==a][:,1]
                    neigh_a_types = particle_ids[neigh_a]
                    neigh_a = neigh_a[(neigh_a_types!=self.radical_thiol)|(neigh_a_types!=self.radical_carbon)]
                    bond_ids,bond_counts = np.unique((snapshot.bonds.group).flatten(),return_counts=True)
                    neigh_a = neigh_a[bond_counts[neigh_a]==1]
                    
                    react = np.random.uniform()
                    if len(neigh_a)>0 and react <= self.thiol_reaction_probability:  
                        b = np.random.choice(neigh_a)  
                    
                        bonds_on_b_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==b]
                        bonds_on_b_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==b]
                        bonds_on_b = np.unique(np.vstack((bonds_on_b_1,bonds_on_b_2)).flatten())
                        bonds_on_b = bonds_on_b[bonds_on_b!=b]

                        bonds_on_a_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==a]
                        bonds_on_a_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==a]
                        bonds_on_a = np.unique(np.vstack((bonds_on_a_1,bonds_on_a_2)).flatten())
                        bonds_on_a = bonds_on_a[bonds_on_a!=a]
                        
                        if self.simple_system == False: 
                            U = np.where(snapshot.angles.typeid==self.dummy_type_angle)[0][0]
                            # good
                            snapshot.angles.group[U]=[bonds_on_b[0],b,a]  # angle 
                            snapshot.angles.typeid[U]=1
                            # good
                            snapshot.angles.group[U+1]=[bonds_on_a[0],a,b] #angle 
                            snapshot.angles.typeid[U+1]=1

                        # flip types of bonds 
                        snapshot.bonds.typeid[snapshot.bonds.group[:,0]==a]=3
                        snapshot.bonds.typeid[snapshot.bonds.group[:,1]==a]=3
                        snapshot.bonds.typeid[snapshot.bonds.group[:,0]==b]=4
                        snapshot.bonds.typeid[snapshot.bonds.group[:,1]==b]=4

                        # make new bond 
                        U = np.where(snapshot.bonds.typeid==self.dummy_type_bond)[0][0]
                        snapshot.bonds.group[U]=[a,b]  # propagation 
                        snapshot.bonds.typeid[U]=1

                        snapshot.particles.typeid[idx[a]]=self.thiol_type  # flip back to Thiol
                        snapshot.particles.typeid[idx[b]]=self.ene_type   
                        snapshot.particles.typeid[idx[bonds_on_b[0]]]=self.radical_carbon   # turn the B into a radical 
                        
                        particle_ids[a]=self.thiol_type
                        particle_ids[b]=self.ene_type
                        particle_ids[bonds_on_b[0]]=self.radical_carbon
                     


    def chain_transfer(self,snapshot,box,positions,particle_ids,ids,idx):
        
         # transfer carbon radical to new thiol 
        if self.chain_transfer_probability>0:
            radicals_positions = positions[particle_ids==self.radical_carbon]
            radicals_ids = ids[particle_ids==self.radical_carbon]
            polymer_ids = ids[particle_ids==self.thiol_type]  # all Thiols

            bond_ids,bond_counts = np.unique((snapshot.bonds.group).flatten(),return_counts=True)
            only_one_bond_ids = bond_ids[bond_counts==1]
            polymer_candidates_ids = np.intersect1d(only_one_bond_ids,polymer_ids)
            polymer_candidates_positions = positions[polymer_candidates_ids]

            if len(radicals_ids)>0:

                aq = freud.locality.AABBQuery(box, radicals_positions)
                nlist = aq.query(polymer_candidates_positions, {'r_max': self.rcut_neigh, 'exclude_ii':True}).toNeighborList()
                all_pairs = nlist[:]

                actual_ids_polymers = polymer_candidates_ids[all_pairs[:,0]]
                actual_ids_radicals = radicals_ids[all_pairs[:,1]]
                actual_ids = np.vstack((actual_ids_radicals,actual_ids_polymers)).T  

                for a in np.unique(actual_ids_radicals):  # chain transfer 
                    
                    neigh_a = actual_ids[actual_ids[:,0]==a][:,1]
                    neigh_a_types = particle_ids[neigh_a]
                    
                    neigh_a = neigh_a[(neigh_a_types!=self.radical_thiol)|(neigh_a_types!=self.radical_carbon)]
                    bond_ids,bond_counts = np.unique((snapshot.bonds.group).flatten(),return_counts=True)
                    neigh_a = neigh_a[bond_counts[neigh_a]==1]

                    transfer = np.random.uniform()
                    if len(neigh_a)>0 and transfer <= self.chain_transfer_probability:  
                        b = np.random.choice(neigh_a)  
                    
                        snapshot.particles.typeid[idx[a]]=self.ene_type  # flip back to Ene
                        snapshot.particles.typeid[idx[b]]=self.radical_thiol  # chain transfer 
                    
                        particle_ids[a]=self.ene_type
                        particle_ids[b]=self.radical_thiol



    def chain_growth(self,snapshot,box,positions,particle_ids,ids,idx):

         # chain side reaction radical carbon with ene reaction                
        if self.chain_side_reaction_probability>0:
            radicals_positions = positions[particle_ids==self.radical_carbon]
            radicals_ids = ids[particle_ids==self.radical_carbon]
            polymer_ids = ids[particle_ids==self.ene_type]   # all carbon/enes

            # bond type = 2 unreacted ene = carbon double bond. 
            # bond type = 4 reacted ene =  carbon single bond - can't react in chain growth 
            # IF IM RIGHT IN THE BOND FORMATION FUNCTION, THIS SHOULD ALSO INCLUDE 4
            bond_ids,bond_counts = np.unique((snapshot.bonds.group[(snapshot.bonds.typeid==2) | (snapshot.bonds.typeid==4)]).flatten(),return_counts=True)
            only_one_bond_ids = bond_ids[bond_counts==1]
            polymer_candidates_ids = np.intersect1d(only_one_bond_ids,polymer_ids)
            polymer_candidates_positions = positions[polymer_candidates_ids]

            if len(radicals_ids)>0:

                aq = freud.locality.AABBQuery(box, radicals_positions)
                nlist = aq.query(polymer_candidates_positions, {'r_max': self.rcut_neigh, 'exclude_ii':True}).toNeighborList()
                all_pairs = nlist[:]

                actual_ids_polymers = polymer_candidates_ids[all_pairs[:,0]]
                actual_ids_radicals = radicals_ids[all_pairs[:,1]]
                actual_ids = np.vstack((actual_ids_radicals,actual_ids_polymers)).T  

                for a in np.unique(actual_ids_radicals):  # radical carbon -ene bond formation 
                    
                    neigh_a = actual_ids[actual_ids[:,0]==a][:,1]
                    neigh_a_types = particle_ids[neigh_a]
                    
                    neigh_a = neigh_a[(neigh_a_types!=self.radical_thiol)|(neigh_a_types!=self.radical_carbon)]
                    bond_ids,bond_counts = np.unique((snapshot.bonds.group).flatten(),return_counts=True)
                    neigh_a = neigh_a[bond_counts[neigh_a]==1]

                    react = np.random.uniform()
                    if len(neigh_a)>0 and react <= self.chain_side_reaction_probability:  
                        b = np.random.choice(neigh_a)  
                    
                        bonds_on_b_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==b]
                        bonds_on_b_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==b]
                        bonds_on_b = np.unique(np.vstack((bonds_on_b_1,bonds_on_b_2)).flatten())
                        bonds_on_b = bonds_on_b[bonds_on_b!=b]

                        bonds_on_a_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==a]
                        bonds_on_a_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==a]
                        bonds_on_a = np.unique(np.vstack((bonds_on_a_1,bonds_on_a_2)).flatten())
                        bonds_on_a = bonds_on_a[bonds_on_a!=a]
                        
                        if self.simple_system == False: 
                            U = np.where(snapshot.angles.typeid==self.dummy_type_angle)[0][0]
                            
                            snapshot.angles.group[U]=[bonds_on_b[0],b,a]  # angle 
                            snapshot.angles.typeid[U]=1
                           
                            snapshot.angles.group[U+1]=[bonds_on_a[0],a,b] #angle 
                            snapshot.angles.typeid[U+1]=1

                        
                        # flip types of bonds 
                        snapshot.bonds.typeid[snapshot.bonds.group[:,0]==a]=4
                        snapshot.bonds.typeid[snapshot.bonds.group[:,1]==a]=4
                        snapshot.bonds.typeid[snapshot.bonds.group[:,0]==b]=4
                        snapshot.bonds.typeid[snapshot.bonds.group[:,1]==b]=4

                        U = np.where(snapshot.bonds.typeid==self.dummy_type_bond)[0][0]
                        snapshot.bonds.group[U]=[a,b]  # propagation 
                        snapshot.bonds.typeid[U]=1
                    
                        snapshot.particles.typeid[idx[a]]=self.ene_type  # flip back to ene
                        snapshot.particles.typeid[idx[b]]=self.ene_type   
                        snapshot.particles.typeid[idx[bonds_on_b[0]]]=self.radical_carbon   # turn the B into a radical 
                        
                        particle_ids[a]=self.ene_type
                        particle_ids[b]=self.ene_type
                        particle_ids[bonds_on_b[0]]=self.radical_carbon



    def termination_reactions(self,snapshot,box,positions,particle_ids,ids,idx):    

        # radical- radical reaction 
        # TODO: does it matter where the thiol group is? i.e. its bonding state? 
        radicals_positions = positions[(particle_ids==self.radical_thiol) | (particle_ids==self.radical_carbon)]
        radicals_ids = ids[(particle_ids==self.radical_thiol) | (particle_ids==self.radical_carbon)]

        if len(radicals_ids)>0:
            aq = freud.locality.AABBQuery(box, radicals_positions)
            nlist = aq.query(radicals_positions, {'r_max': self.rcut_neigh, 'exclude_ii':True}).toNeighborList()
            all_pairs = nlist[:]

            actual_ids_polymers = radicals_ids[all_pairs[:,0]]
            actual_ids_radicals = radicals_ids[all_pairs[:,1]]
            actual_ids = np.vstack((actual_ids_radicals,actual_ids_polymers)).T
            # sorting and making it unique to not make same bonds again 
            actual_ids = np.sort(actual_ids,axis=1)
            actual_ids = np.unique(actual_ids,axis=0)
            

            for a in np.unique(actual_ids[:,0]):

                neigh_a = actual_ids[actual_ids[:,0]==a][:,1]

                if len(neigh_a)>0:
                    b = np.random.choice(neigh_a)  
                   
                    bonds_on_b_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==b]
                    bonds_on_b_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==b]
                    bonds_on_b = np.unique(np.vstack((bonds_on_b_1,bonds_on_b_2)).flatten())
                    bonds_on_b = bonds_on_b[bonds_on_b!=b]

                    bonds_on_a_1 = snapshot.bonds.group[snapshot.bonds.group[:,0]==a]
                    bonds_on_a_2 = snapshot.bonds.group[snapshot.bonds.group[:,1]==a]
                    bonds_on_a = np.unique(np.vstack((bonds_on_a_1,bonds_on_a_2)).flatten())
                    bonds_on_a = bonds_on_a[bonds_on_a!=a]
                    
                    if self.simple_system == False: 
                        U = np.where(snapshot.angles.typeid==self.dummy_type_angle)[0][0]
                        # good
                        snapshot.angles.group[U]=[bonds_on_b[0],b,a]  # angle 
                        snapshot.angles.typeid[U]=1
                        # good
                        snapshot.angles.group[U+1]=[bonds_on_a[0],a,b] #angle 
                        snapshot.angles.typeid[U+1]=1

                    # good 
                    U = np.where(snapshot.bonds.typeid==self.dummy_type_bond)[0][0]
                    snapshot.bonds.group[U]=[a,b]  # propagation 
                    snapshot.bonds.typeid[U]=1

                    print("radical anhiliation!")
                    print(a,b)
                    print(particle_ids[a],particle_ids[b])

                    # radicals anhiliate each other - both flip back
                    if  snapshot.particles.typeid[idx[a]]==self.radical_thiol:
                        snapshot.particles.typeid[idx[a]]=self.thiol_type  
                        particle_ids[a]=self.thiol_type

                    if  snapshot.particles.typeid[idx[b]]==self.radical_thiol:
                        snapshot.particles.typeid[idx[b]]=self.thiol_type   
                        particle_ids[b]=self.thiol_type 

                    if  snapshot.particles.typeid[idx[a]]==self.radical_carbon:
                        snapshot.particles.typeid[idx[a]]=self.ene_type  
                        particle_ids[a]=self.ene_type
                        
                    if  snapshot.particles.typeid[idx[b]]==self.radical_carbon:
                        snapshot.particles.typeid[idx[b]]=self.ene_type   
                        particle_ids[b]=self.ene_type 

                    print(particle_ids[a],particle_ids[b])

                   
    def propagate_reaction(self,
                           snapshot,r_cut=1.1,
                           chain_transfer_probability=1.0,
                           chain_side_reaction_probability=0.0,
                           thiol_reaction_probability=1.0,
                           is_cpu=True):
         
        self.chain_transfer_probability=chain_transfer_probability
        self.chain_side_reaction_probability=chain_side_reaction_probability
        self.thiol_reaction_probability = thiol_reaction_probability 
        self.rcut_neigh = r_cut 

        box = freud.Box.from_box(snapshot.global_box)
        ids = np.arange(len(snapshot.particles.tag))
        if is_cpu:
            idx = snapshot.particles.rtag[ids]
            particle_ids = snapshot.particles.typeid[idx]
            positions = snapshot.particles.position[idx]
        else:
            idx = cupy.array(snapshot.particles.rtag,copy=True)[ids]
            particle_ids = cupy.array(snapshot.particles.typeid,copy=True)[idx]
            positions = cupy.array(snapshot.particles.position,copy=True)[idx]
        
        # radical radical_thiol with Ene = Propagation/bond formation step 
        self.bond_formation(snapshot,box,positions,particle_ids,ids,idx)
                    
        # reaction radical_carbon with Thiol  - chain transfer step
        self.chain_transfer(snapshot,box,positions,particle_ids,ids,idx)

        #  competing chain growth radical carbond with ene reaction 
        self.chain_growth(snapshot,box,positions,particle_ids,ids,idx)

        # termination reactions, i.e radical-radical interactions 
        # self.termination_reactions(snapshot,box,positions,particle_ids,ids,idx)


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
