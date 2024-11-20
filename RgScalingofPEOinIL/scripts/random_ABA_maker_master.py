#Initializes a cylindrical field oriented in the z direction

import numpy as np
import os,sys
import gsd.hoomd
# sys.path.insert(0,'/home/bj21/programs/hoomd-blue-3-DHOOMD_GPU_PLATFORM=CUDA-DBUILD_HPMC=OFF')
# import hoomd
from hoomd.data import make_snapshot, boxdim
import sys, math

def random_point_on_cone(R,theta,prev):
    """ Returns random vector with length R and angle theta between previous one, uniformly distributed """
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

def random_FRC(m,characteristic_ratio,bond_length):
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
        # this line below has been modified on 03312021 to ensure the angle theta is representing the bond angle theta not tau
	# Set the c = 3.3 for the measurements
        # see http://polymerdatabase.com/polymer%20physics/Freely-Rotating%20Chain.html
        prev = coords[i-1]-coords[i-2]
        n = random_point_on_cone(bond_length,theta,prev)
        new = coords[i-1]+n
        coords[i]=new

    return coords

def cyl_to_cart(theta,r):
    """Converts a point given in cylindrical coordinates into cartesian coordinates"""
    x0 = r*math.cos(theta)
    x1 = r*math.sin(theta)
    return x0, x1

class MPSTriblockMaker():

    def __init__(self,vol_frac,NB,fA,b,c,morphology):
        self.NB = NB
        self.vol_frac = vol_frac
        self.fA = fA
        self.bond = b
        self.c = c
        self.morphology = morphology

        self.characterize_system()

        #count the number of field particles that will be added, depending on the
        #   morphology
        self.num_field_particles = 0
        match fA:
            case 0.16:
                # spherical morphology
                self.sphere_centers = self.find_sphere_centers()
                self.num_field_layers = int(self.spacing/2)
                # count the number of field particles that will be added. Necessary for
                #   set_types not to fail
                for center in self.sphere_centers:
                    for r,j in enumerate(range(self.num_field_layers)):
                        if r == 0:
                            points = center
                            n = 10
                            self.num_field_particles += 1
                        else:
                            points = self.points_on_sphere(center=center,r=r,n=n)
                            n = n * ( ( ( (r+1)/r )**2 - 1 ) * 0.5 + 1 )
                            self.num_field_particles = self.num_field_particles + len(points)
            case 0.26:
                #cylindrical morphology
                self.cylinder_centers = self.find_cylinder_centers()
                self.n_field_planes = 7
                step_arr = np.linspace(4,4,num=self.n_field_planes)
                for center in self.cylinder_centers:
                    for r,j in enumerate(range(self.n_field_planes)):
                        if r == 0:
                            for z in np.arange(-self.Lz/2,self.Lz/2,step=1):
                                self.num_field_particles += 1
                        else:
                            # The cylinder is not just a line
                            # angle_step = 2*math.pi/int(2*math.pi*r/step_arr[j]) #radians
                            angle_step = 2*math.pi/step_arr[j] #radians
                            for z in np.arange(-self.Lz/2,self.Lz/2,step=1):
                                for theta in np.arange(0,2*math.pi,step=angle_step):
                                    self.num_field_particles += 1
            case 0.5:
                #lamellar morphology, diagonal or not
                Nlamellae = 8
                self.field_width = self.Lx/Nlamellae
                self.n_field_planes = int(self.field_width)
                self.field_step = self.field_width/int(self.field_width)
                steps_arr = np.linspace(1,5,num=self.n_field_planes)
                steps_arr = np.append(steps_arr,np.linspace(5,1,num=self.n_field_planes))
                for index,i in enumerate(np.arange(-self.Lx/2,self.Lx/2,step=self.field_step)):
                    step = steps_arr[index%(self.n_field_planes*2)]
                    for j in np.arange(-self.Ly/2,self.Ly/2,step=step):
                        for k in np.arange(-self.Lz/2,self.Lz/2,step=step):
                            self.num_field_particles = self.num_field_particles + 1
        # Now have the number of field particles that will be added in self.num_field_particles

        self.N = int(self.num_pol*self.num_mon+self.num_field_particles)

        self.positions  = []
        # set types, bonds, positions
        self.set_types()
        self.set_bonds()
        self.set_positions()

    def characterize_system(self):
        # model parameters - right now no SP, masses are hardcoded to be (1, 1.6)
        self.bAA = 0.9609
        self.bAB = 1.0382
        self.bBB = 1.1116
        self.sAA = 1.0
        self.sBB = 1.2
        self.sAB = 0.5*(self.sAA+self.sBB)

        # Find the number of A and B monomers based on their volumes
        fB = 1-self.fA
        A_mon_volume = 4*np.pi/3.0*((self.sAA/2.)**3)
        B_mon_volume = 4*np.pi/3.0*((self.sBB/2.)**3)
        B_chain_volume = B_mon_volume*self.NB
        chain_volume = B_chain_volume/fB
        A_chain_volume = chain_volume*self.fA
        self.NA = round(A_chain_volume/A_mon_volume/2)
        self.num_mon = self.NA*2+self.NB
        chain_length = self.NA*2 + self.NB

        #Find the box dimensions and set them according to the defined morphology
        self.Lx,self.Ly,self.Lz = self.find_box(self.fA)
        self.box = boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz)

        n_chains = int(np.round((self.Lx*self.Ly*self.Lz)*self.vol_frac / chain_volume))
        self.num_pol = n_chains

        #Check the actual volume fraction and print system characteristics
        nA=(self.NA*2*n_chains*(self.sAA/2.)**3*4*np.pi/3.0)/(self.Lx*self.Ly*self.Lz)     # Volume fraction of A
        nB=(self.NB*n_chains*(self.sBB/2.)**3*4*np.pi/3.0)/(self.Lx*self.Ly*self.Lz)       # Volume fraction of B
        print("volume fractions A = %f, B= %f"%(nA,nB))
        print("volume fractions A = %f, B= %f"%(self.NA*2*(self.sAA/2.)**3/(self.NA*2*(self.sAA/2.)**3+self.NB*(self.sBB/2.)**3),1-self.NA*2*(self.sAA/2.)**3/(self.NA*2*(self.sAA/2.)**3+self.NB*(self.sBB/2.)**3)))
        print("Number of polymers %d in box %2.1f x  %2.1f x %2.1f"%(n_chains,self.Lx,self.Ly,self.Lz))
        print("Polymer number f_A = %1.2f N_A= %d N_B= %d , N= %d, 2*N = %d"%(self.fA,self.NA,self.NB,chain_length/2,chain_length))

    def find_box(self,fA):
        """
        Based on the user-specified volume fraction of A, determines the morphology to form
            and returns the box dimensions
        """
        match fA:
            case 0.16:
                #spherical morphology
                if self.morphology != "spherical":
                    print(f"""ERROR - Morphology {self.morphology} and volume fraction of 
                          A {fA} is inconsistent!""")
                    exit(2)

                # find the spacing between spheres based on Matsen and Thompson 1999 SCFT calculations
                a = self.sAB #statistical segment length assumed 1.1 as the average between A and B monomers
                N = self.num_mon
                print(N)
                D_prime = 1.32*a*N**0.5 # Reading from Figure 7b at f_A = 0.16
                print(D_prime)
                self.spacing = (1.5)**0.5*D_prime
                print("spherical spacing: ",self.spacing)
                #make the box match the box size
                Lx = 4*self.spacing
                Ly = 4*self.spacing
                Lz = 4*self.spacing
                return Lx,Ly,Lz
            case 0.26:
                #cylindrical morphology
                if self.morphology != "cylindrical":
                    print(f"""ERROR - Morphology {self.morphology} and volume fraction of 
                          A {fA} is inconsistent!""")
                    exit(2)

                # find the spacing between spheres based on Matsen and Thompson 1999 SCFT calculations
                a = self.sAB #statistical segment length assumed 1.1 as the average between A and B monomers
                N = self.num_mon
                D_prime = 1.58*a*N**0.5 # Reading from Figure 7b at f_A = 0.25
                self.spacing = (4/3)**0.5*D_prime
                print("cylindrical spacing: ",self.spacing)
                #make the box match the box size
                Lx = 3*self.spacing
                Ly = 2*self.spacing*3**0.5
                Lz = 40
                return Lx,Ly,Lz
            case 0.5:
                #lamellar morphology
                if self.morphology != "lamellar" and self.morphology != "diagonal_lamellar" :
                    print(f"""ERROR - Morphology {self.morphology} and volume fraction of 
                          A {fA} is inconsistent!""")
                    exit(2)

                a = self.sAB #statistical segment length assumed 1.1 as the average between A and B monomers
                N = self.num_mon
                D_prime = 1.9*a*N**0.5 # Reading from Figure 7b at f_A = 0.5
                self.spacing = D_prime
                 
                Lx = 4*self.spacing
                Ly = 3*self.spacing
                Lz = 43
                #original diagonal lamellar
                # Lx = 104
                # Ly = 78
                # Lz = 43
                #original lamellar
                # Lx = 102
                # Ly = 88
                # Lz = 43
                return Lx,Ly,Lz

    def find_sphere_centers(self):
        sphere_centers = np.empty((0,3))
        #Build the spherical domain centers by considering two cubic grids
        # https://stackoverflow.com/questions/68118298/finding-all-possible-combinations-of-multiple-arrays-where-all-combinations-als
        from itertools import product
        # 1st cubic grid
        sphere_centers_grid_1 = np.arange(0,self.Lx,step=self.spacing)
        for point in product(*[sphere_centers_grid_1,sphere_centers_grid_1,sphere_centers_grid_1]):
            sphere_centers = np.vstack((sphere_centers,point))
        # print(sphere_centers_grid_1)
        #2nd cubic grid
        sphere_centers_grid_2 = np.add(sphere_centers_grid_1,self.spacing/2)
        i = 0
        for point in product(*[sphere_centers_grid_2,sphere_centers_grid_2,sphere_centers_grid_2]):
            sphere_centers = np.vstack((sphere_centers,point))
        # print(sphere_centers_grid_2)
        sphere_centers = np.add(sphere_centers,-self.Lx/2)
        return sphere_centers
    
    def find_cylinder_centers(self):
        cylinder_centers = np.empty((0,2))
        #Build the spherical domain centers by considering two cubic grids
        # https://stackoverflow.com/questions/68118298/finding-all-possible-combinations-of-multiple-arrays-where-all-combinations-als
        from itertools import product
        # Layer A
        cylinder_centers_layer_A = np.arange(0,self.Lx,step=self.spacing)
        cylinder_centers_layer_A_pos = np.arange(0,self.Ly,step=self.spacing*3**0.5)
        for point in product(*[cylinder_centers_layer_A,cylinder_centers_layer_A_pos]):
            cylinder_centers = np.vstack((cylinder_centers,point))
        # Layer B
        cylinder_centers_layer_B = np.add(cylinder_centers_layer_A,self.spacing/2)
        cylinder_centers_layer_B_pos = np.add(cylinder_centers_layer_A_pos,self.spacing/2*3**0.5)
        i = 0
        for point in product(*[cylinder_centers_layer_B,cylinder_centers_layer_B_pos]):
            cylinder_centers = np.vstack((cylinder_centers,point))
        # shift to hoomd cell coordinate origin
        cylinder_centers[:,0] = np.add(cylinder_centers[:,0],-self.Lx/2)
        cylinder_centers[:,1] = np.add(cylinder_centers[:,1],-self.Ly/2)
        return cylinder_centers

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

        # -------------- set the field particle positions ---------------
        match self.morphology:
            case "spherical":
                #add spherical field
                for center in self.sphere_centers:
                    for r,j in enumerate(range(self.num_field_layers)):
                        if r == 0:
                            points = center
                            n = 10
                        else:
                            points = self.points_on_sphere(center=center,r=r,n=n)
                            n = n * ( ( ( (r+1)/r )**2 - 1 ) * 0.5 + 1 )
                        # print('n:',n)
                        points = self.wrap_pbc(points)
                        self.positions = np.vstack((self.positions,points))
                        # self.positions = np.vstack((self.positions,center))
            case "cylindrical":
                #add cylindrical field
                #steps will be the size of the steps taken on the target face of the cylinder,
                #   both in the z direction and around the circumference
                step_arr = np.linspace(4,4,num=self.n_field_planes)
                # step_arr = np.append(step_arr,np.linspace(5,1,num=self.n_field_planes))
                for center in self.cylinder_centers:
                    for r,j in enumerate(range(self.n_field_planes)):
                        points = self.points_on_cylinder(center=center,r=r,step=step_arr[j])
                        points = self.wrap_pbc(points)
                        self.positions = np.vstack((self.positions,points))
            case "lamellar":
                #add lamellar field
                steps_arr = np.linspace(1,5,num=self.n_field_planes)
                steps_arr = np.append(steps_arr,np.linspace(5,1,num=self.n_field_planes))
                for index,i in enumerate(np.arange(-self.Lx/2,self.Lx/2,step=self.field_step)):
                # for index,i in enumerate(np.linspace(-self.Lx/2,self.Lx/2,num=(self.field_width)*self.Nlamellae)):
                    step = steps_arr[index%(self.n_field_planes*2)]
                    y_shift = np.random.rand()*-step
                    z_shift = np.random.rand()*-step
                    for j in np.arange(-self.Ly/2,self.Ly/2,step=step):
                        for k in np.arange(-self.Lz/2,self.Lz/2,step=step):
                            if j == -self.Ly/2:
                                j = self.Ly/2
                            if k == -self.Lz/2:
                                k = self.Lz/2
                            r = [i,j+y_shift,k+z_shift]
                            self.positions = np.vstack((self.positions, r))
            case "diagonal_lamellar":
                #add lamellar field
                steps_arr = np.linspace(1,5,num=self.n_field_planes)
                steps_arr = np.append(steps_arr,np.linspace(5,1,num=self.n_field_planes))
                for index,i in enumerate(np.arange(-self.Lx/2,self.Lx/2,step=self.field_step)):
                # for index,i in enumerate(np.linspace(-self.Lx/2,self.Lx/2,num=(self.field_width)*self.Nlamellae)):
                    step = steps_arr[index%(self.n_field_planes*2)]
                    y_shift = np.random.rand()*-step
                    z_shift = np.random.rand()*-step
                    for j in np.arange(-self.Ly/2,self.Ly/2,step=step):
                        for k in np.arange(-self.Lz/2,self.Lz/2,step=step):
                            x_shift = j
                            x_position = i + x_shift
                            if j == -self.Ly/2:
                                j = self.Ly/2
                            if k == -self.Lz/2:
                                k = self.Lz/2
                            r = [x_position,j+y_shift,k+z_shift]
                            self.positions = np.vstack((self.positions, r))

    def points_on_sphere(self,center,r,n):
        flip = np.random.rand(3)
        for i in range(len(flip)):
            if flip[i] > 0.5:
                flip[i] = -1
            else:
                flip[i] = 1
        # print(flip)
        goldenRatio = (1 + 5**0.5)/2
        iterator = np.arange(0, n)
        theta = 2 *np.pi * iterator / goldenRatio
        phi = np.arccos(1 - 2*(iterator)/n)
        x, y, z = center[0] + flip[0] * r * np.cos(theta) * np.sin(phi), center[1] + flip[1] * r * np.sin(theta) * np.sin(phi), center[2] + flip[2] * r * np.cos(phi)
        points = np.transpose(np.vstack((x,y,z)))
        return points
    
    def points_on_cylinder(self,center,r,step):
        points = np.array([]).reshape(0,3)
        #cylinder orientation dimension is z
        if r == 0:
            for z in np.arange(-self.Lz/2,self.Lz/2,step=1):
                points = np.vstack((points,[center[0],center[1],z]))
        else:
            # The cylinder is not just a line
            # angle_step = 2*math.pi/int(2*math.pi*r/step) #radians
            angle_step = 2*math.pi/step #radians
            for z in np.arange(-self.Lz/2,self.Lz/2,step=1):
                angle_shift = np.random.rand()*-angle_step
                for theta in np.arange(0,2*math.pi,step=angle_step):
                    x0, x1 = cyl_to_cart(theta+angle_shift,r)
                    points = np.vstack((points,[x0+center[0],x1+center[1],z]))
        return points

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

    def wrap_pbc(self,x):
        box = np.array([self.Lx,self.Ly,self.Lz])
        delta = np.where(x > 0.5 * box, x - box, x)
        delta = np.where(delta < - 0.5 * box, box + delta, delta)
        return delta

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
