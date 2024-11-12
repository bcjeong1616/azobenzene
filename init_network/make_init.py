
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
                 length_crosslinkers,length_sidechains,azo_probability,num_ionic_liquid):

        self.L = L
        self.length_backbone = length_backbone
        self.num_backbone = num_backbone
        self.crosslinking_probability = crosslinking_probability
        self.length_crosslinkers = length_crosslinkers
        self.length_sidechains = length_sidechains
        self.azo_probability = azo_probability
        self.num_ionic_liquid = num_ionic_liquid

        self.bonds = []
        self.positions_backbone = []
        self.positions_crosslinks = []
        self.types_backones = []
        self.types_crosslinkers = []

        self.types_ionic_liquid = []
        self.positions_ionic_liquid = []


    def create_system(self):
        self.create_polymer_system()
        self.add_ionic_iquid()

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
                self.types_backones.append(0)
                if j<len(pos)-1:
                    self.bonds.append([i,i+1])
                i = i + 1

        # sidechains and crosslinks
        for k,p in enumerate(self.positions_backbone):
            # only the ones with id=0 are open
            if self.types_backones[k] == 0:
                # make a crosslink, PEG diacrylate (not azobenze for now)
                random_number = np.random.uniform(0,1)
                if random_number < self.crosslinking_probability:
                    self.types_backones[k] = 1
                    start_pos = p
                    # selection function, returns id of another backbone monomer with no crosslinker on it
                    end_pos_id = self.select_crosslinker(k,self.types_backones,self.positions_backbone)
                    self.types_backones[end_pos_id] = 1

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
                            self.bonds.append([k,i])
                            self.types_crosslinkers.append(0)
                        if j<len(pos)-2:
                            self.bonds.append([i,i+1])
                            self.types_crosslinkers.append(2)
                        if j == len(pos)-2:
                            self.bonds.append([i,i+1])
                            self.types_crosslinkers.append(0)
                        if j == len(pos)-1:
                            self.bonds.append([i,end_pos_id])

                        i = i + 1

                # add sidechain (could be PEG or Azo)
                else:
                    random_number = np.random.uniform(0,1)
                     # add EMIM-Tethered Azo sidechain
                    if random_number < self.azo_probability:
                        self.types_backones[k] = 1
                        start_pos = p

                        length_azo = 17
                        q = self.random_FRC(length_azo+2,0.2,1.0)
                        pos = (start_pos + q[2:])

                        for j,pi in enumerate(pos):
                            p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                            self.positions_crosslinks.append(p_wrap)
                            # by far the least elegant way of doing this. but it works.
                            #  0      1     2       3        4    5      6     7      8    9
                            #['SN4a','TC1','SN3r','SN3r1','SC1','TN3r','TC5','TN2q','TC6','10']
                            if j == 0:
                                self.bonds.append([k,i])
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(0)
                            if j == 1:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(4)
                            if j == 2:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(4)
                            if j == 3:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(5)
                            if j == 4:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(6)
                            if j == 5:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(6)
                            if j == 6:
                                self.bonds.append([i,i+1])
                                self.bonds.append([i,i-2])
                                self.types_crosslinkers.append(6)
                            if j == 7:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(5)
                            if j == 8:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(6)
                            if j == 9:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(6)
                            if j == 10:
                                self.bonds.append([i,i+1])
                                self.bonds.append([i,i-2])
                                self.types_crosslinkers.append(6)
                            if j == 11:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(5)
                            if j == 12:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(4)
                            if j == 13:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(4)
                            if j == 14:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(8)
                            if j == 15:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(7)
                            if j == 16:
                                self.bonds.append([i,i-2])
                                self.types_crosslinkers.append(7)
                            i = i+1

                    # add PEG acrylate sidechain
                    else:
                        self.types_backones[k] = 1
                        start_pos = p
                        # w = self.get_gaussian_vector(self.length_sidechains)
                        # pos = self.random_from_to(1,self.length_sidechains+2,start_pos,start_pos-w)
                        # pos = np.asarray(pos[0][1:-1])

                        q = self.random_FRC(self.length_sidechains+2,0.2,1.0)

                        pos = (start_pos + q[2:])

                        for j,pi in enumerate(pos):
                            p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                            self.positions_crosslinks.append(p_wrap)
                            if j == 0:
                                self.bonds.append([k,i])
                                self.types_crosslinkers.append(0)
                            if j<len(pos)-2:
                                self.bonds.append([i,i+1])
                                self.types_crosslinkers.append(3)
                            if j==len(pos)-2:
                                self.types_crosslinkers.append(3)

                            i = i + 1


        self.positions_crosslinks = np.asarray(self.positions_crosslinks)
        self.positions_backbone = np.asarray(self.positions_backbone)

        #self.bonds = np.asarray(self.bonds)
        self.types_backones = np.asarray(self.types_backones)
        self.types_crosslinkers = np.asarray(self.types_crosslinkers)

    def add_ionic_iquid(self):
        # existing thethered EMIM
        N_current_charge = int(len(self.types_crosslinkers[self.types_crosslinkers==7])/2.0)
        N_to_add_EMIM = int(self.num_ionic_liquid/2.0) - N_current_charge
        #TFSI
        N_to_add_TFSI = int(self.num_ionic_liquid/2.0)
        i = len(self.types_backones) + len(self.types_crosslinkers)

        # 9     10
        #'SX4e','SQ4n'
        for n in range(N_to_add_TFSI):

            self.bonds.append([i,i+1])
            self.bonds.append([i+1,i+2])
            self.bonds.append([i+2,i+3])

            self.types_ionic_liquid.append(9)
            self.types_ionic_liquid.append(10)
            self.types_ionic_liquid.append(10)
            self.types_ionic_liquid.append(9)

            q = self.random_FRC(4,0.2,1.0)
            com = np.random.uniform(-self.L/2.0,self.L/2.0,3)
            pos = com + q

            for j,pi in enumerate(pos):
                p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                self.positions_ionic_liquid.append(p_wrap)

            i = i + 4

        # 7     8
        #'TN2q','TC6'
        for n in range(N_to_add_EMIM):

            self.bonds.append([i,i+1])
            self.bonds.append([i+1,i+2])
            self.bonds.append([i,i+3])
            self.bonds.append([i+1,i+3])

            self.types_ionic_liquid.append(7)
            self.types_ionic_liquid.append(7)
            self.types_ionic_liquid.append(8)
            self.types_ionic_liquid.append(8)

            q = self.random_FRC(4,0.2,1.0)
            com = np.random.uniform(-self.L/2.0,self.L/2.0,3)
            pos = com + q

            for j,pi in enumerate(pos):
                p_wrap =self.wrap_pbc(pi,np.array([self.L,self.L,self.L]))
                self.positions_ionic_liquid.append(p_wrap)

            i = i + 4



    def save_system(self,name_gsd):
        self.output = gsd.hoomd.open(name=name_gsd, mode='w')
        frame = gsd.hoomd.Frame()

        self.positions = np.vstack((self.positions_backbone,
                                             self.positions_crosslinks,
                                             self.positions_ionic_liquid
                                            ))
        self.types = np.hstack((self.types_backones,
                                         self.types_crosslinkers,
                                         self.types_ionic_liquid
                                        ))

        frame.particles.N = len(self.positions)

        self.positions = self.wrap_pbc(self.positions,\
                                       np.array([self.L,self.L,self.L]))

        frame.particles.position = self.positions
                                #  0      1     2       3        4    5      6     7      8    9     10
        frame.particles.types = ['SN4a','TC1','SN3r','SN3r1','SC1','TN3r','TC5','TN2q','TC6','SX4e','SQ4n']
        frame.particles.typeid = self.types

        frame.bonds.N = len(self.bonds)
        frame.bonds.types = ['A','B','C','D']
        frame.bonds.group = self.bonds

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
        coords[1]=[1,0,0]
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
        return res

    def select_crosslinker(self,a,cr,pos):
        pos = np.asarray(pos)
        pos = self.wrap_pbc(pos,np.array([self.L,self.L,self.L]))
        cr = np.asarray(cr)
        picked = False
        max_trials = 1000
        trials = 0
        while picked == False and trials <= max_trials:
            open_crosslinks = np.asarray(np.where(cr<1)[0])
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


