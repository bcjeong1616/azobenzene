import sys
import numpy as np
import hoomd
import math
import gsd.hoomd
import csv
# from martini3 import q_adj
# from martini3 import particles

def quartic_bond(r,k4,b1,U0,sigma,epsilon,r_cut):
    # hardcoding delta_r = r_cut
    delta_r = r_cut
    # hardcoding b1 and b2 based on k4 to get same eq length
    if (math.isclose(sigma, 1, abs_tol=0.001)):
        req = 0.96
    else:
        if (math.isclose(sigma, 1.2, abs_tol=0.001)):
            req = 1.131
        else:
            if (math.isclose(sigma, 1.1, abs_tol=0.001)):
                req = 1.0465
            else:
                print(f"All cases failed! sigma: {sigma}")
    # b2 = 0
    b2 = (-30*req/(1-req**2/r_cut**2)-3*b1*k4*(req-r_cut)**2+4*k4*(req-r_cut)**3)/(-2*b1*k4*(req-r_cut)+3*k4*(req-r_cut)**2)
    # b1 = (-30*req/(1-req**2/r_cut**2)+4*k4*(req-r_cut)**3)/(3*k4*(req-r_cut)**2)
    conds = [r < r_cut, r >= r_cut]
    lj = LJ_potential(r,sigma,epsilon,sigma*2**(1/6))
    eps = np.piecewise(r,  [r < sigma*2**(1/6), r >= sigma*2**(1/6)],\
                           [lambda x: epsilon , lambda x: 0])
    funcs = [lambda x: k4*(x-delta_r-b1)*(x-delta_r-b2)*(x-delta_r)**2+U0 , lambda x: U0]
    quartic_potential = np.piecewise(r, conds, funcs)
    return quartic_potential+lj+eps

def LJ_potential(r,sigma,epsilon,r_cut):
    conds = [r < r_cut, r >= r_cut]
    funcs = [lambda x: 4*epsilon*((sigma/x)**(12)-(sigma/x)**6), lambda x: 0]
    return np.piecewise(r, conds, funcs)

def FENE_potential(r,sigma,epsilon,R,K):
    lj = LJ_potential(r,sigma,epsilon,sigma*2**(1/6))
    eps = np.piecewise(r,  [r < sigma*2**(1/6), r >= sigma*2**(1/6)],[lambda x: epsilon , lambda x: 0])
    bond = -0.5*K*R**2*np.log(1-(r/R)**2) + lj + eps
    return bond


class PotentialParameterizer():
    def __init__(self):
        # model parameters - masses are hardcoded in the polymerMelt Maker
        self.bPEOPEO = 0.9609
        self.bPEOBenz = 0.9609
        self.bBenzNN = 0.9609
        self.bPEOEndgroup = 0.9609
        self.sPEOPEO = 1.0
        self.sBenzBenz = 1.0
        self.sNNNN = 1.0
        self.sEndgroupEndgroup = 1.0
        self.sPEOBenz = 0.5*(self.sPEOPEO+self.sBenzBenz)
        self.sPEONN = 0.5*(self.sPEOPEO+self.sNNNN)
        self.sPEOEndgroup = 0.5*(self.sPEOPEO+self.sEndgroupEndgroup)
        self.sBenzNN = 0.5*(self.sBenzBenz+self.sNNNN)
        self.sBenzEndgroup = 0.5*(self.sBenzBenz+self.sEndgroupEndgroup)
        self.sNNEndgroup = 0.5*(self.sNNNN+self.sEndgroupEndgroup)

        self.ePEOPEO = 1.0
        self.eBenzBenz = 1.0
        self.eNNNN = 1.0
        self.ePEOBenz = 2*self.ePEOPEO*self.eBenzBenz/(self.ePEOPEO+self.eBenzBenz)
        self.ePEONN = 2*self.ePEOPEO*self.eNNNN/(self.ePEOPEO+self.eNNNN)
        self.eBenzNN = 2*self.eBenzBenz*self.eNNNN/(self.eBenzBenz+self.eNNNN)
        self.sBenzBenz = 1.0

    
    def init_lj_potentials(types, cell):
        """
        initialize the lj potentials of all types of beads in your simulation
        according to the Martini3 specifications

        Args:
        types (list of Particles): List of all particles types in simulation.
        cell (hoomd neighborlist): HOOMD neighborlist

        Returns:
        hoomd.md.pair.LJ: Pair potential between all particle types.
        """
        lj = hoomd.md.pair.LJ(nlist=cell)

        for i in range(len(types)):
            for j in range(len(types)):
                particle_a = types[i]
                particle_b = types[j]

                # Adjust for Q bead interactions if one of the beads is Q
                if (particle_a.lj_params[particle_b.name][0] !=0) and "Q" in (particle_a.name) and "Q" not in (particle_b.name):
                    if q_adj.eps_b.get(particle_b.name) != None:
                        eps_b = q_adj.eps_b.get(particle_b.name)
                        b_size = "R"
                    else:
                        if q_adj.eps_b.get(particle_b.name[0:2]) != None:
                            b_size = "R"
                            eps_b = q_adj.eps_b.get(particle_b.name[0:2])
                        elif q_adj.eps_b.get(particle_b.name[1:3]) != None:
                            b_size = particle_b.name[0]
                            eps_b = q_adj.eps_b.get(particle_b.name[1:3])
                        else:
                            print(
                                "error lipid.py line 78 idk what would cause this case but writing this "
                            )

                    # This line assumes that particle a is a Q bead and not a D bead and that it has no letter modifiers
                    if len(particle_a.name) > 2:
                        a_size = particle_a.name[0]
                        a_name = particle_a.name[1:3]
                    else:
                        a_size = "R"
                        a_name = particle_a.name

                    eps_qb = particle_a.lj_params[particle_b.name][1]
                    eps_w = q_adj.eps_b.get("W")
                    eps_c1 = q_adj.eps_b.get("C1")
                    gamma = q_adj.gamma.get(a_name).get(b_size)
                    eps_inside_w = q_adj.eps_w.get(a_name).get(b_size)
                    eps_inside_c1 = q_adj.eps_c1.get(a_name).get(b_size)
                    p_qb = q_adj.p_qb.get(a_size).get(b_size)

                    sigma_qw = particle_a.lj_params["W"][0]
                    sigma_qb = particle_a.lj_params[particle_b.name][0]

                    eps_b_inside = eps_inside_w + (eps_b - eps_w) / (eps_c1 - eps_w) * (
                        eps_inside_c1 - eps_inside_w
                    )

                    eps_final = (
                        eps_qb
                        + p_qb
                        * gamma
                        * sigma_qw
                        / sigma_qb
                        * eps_w
                        * eps_inside_w
                        / eps_b
                        / eps_b_inside
                        * (eps_b - eps_b_inside)
                        / (eps_w - eps_inside_w)
                    )

                    lj.params[(particle_a.name, particle_b.name)] = dict(
                        epsilon=eps_final, sigma=particle_a.lj_params[particle_b.name][0]
                    )
                    lj.r_cut[(particle_a.name, particle_b.name)] = 1.1  # nm
                elif (particle_a.lj_params[particle_b.name][0] !=0)  and "Q" not in (particle_a.name) and "Q" in (particle_b.name):
                    if q_adj.eps_b.get(particle_a.name) != None:
                        eps_b = q_adj.eps_b.get(particle_a.name)
                        a_size = "R"
                    else:
                        if q_adj.eps_b.get(particle_a.name[0:2]) != None:
                            a_size = "R"
                            eps_b = q_adj.eps_b.get(particle_a.name[0:2])
                        elif q_adj.eps_b.get(particle_a.name[1:3]) != None:
                            a_size = particle_a.name[0]
                            eps_b = q_adj.eps_b.get(particle_a.name[1:3])
                        else:
                            print(
                                "error lipid.py line 78 idk what would cause this case but writing this "
                            )

                    # This line assumes that particle b is a Q bead and not a D bead
                    # this also assumes Q beads have no name modifiers
                    if len(particle_b.name) > 2:
                        b_size = particle_b.name[0]
                        b_name = particle_b.name[1:3]
                    else:
                        b_size = "R"
                        b_name = particle_b.name

                    eps_qb = particle_b.lj_params[particle_a.name][1]
                    eps_w = q_adj.eps_b.get("W")
                    eps_c1 = q_adj.eps_b.get("C1")
                    gamma = q_adj.gamma.get(b_name).get(a_size)
                    eps_inside_w = q_adj.eps_w.get(b_name).get(a_size)
                    eps_inside_c1 = q_adj.eps_c1.get(b_name).get(a_size)
                    p_qb = q_adj.p_qb.get(b_size).get(a_size)

                    sigma_qw = particle_b.lj_params["W"][0]
                    sigma_qb = particle_b.lj_params[particle_a.name][0]

                    eps_b_inside = eps_inside_w + (eps_b - eps_w) / (eps_c1 - eps_w) * (
                        eps_inside_c1 - eps_inside_w
                    )

                    eps_final = (
                        eps_qb
                        + p_qb
                        * gamma
                        * sigma_qw
                        / sigma_qb
                        * eps_w
                        * eps_inside_w
                        / eps_b
                        / eps_b_inside
                        * (eps_b - eps_b_inside)
                        / (eps_w - eps_inside_w)
                    )

                    lj.params[(particle_a.name, particle_b.name)] = dict(
                        epsilon=eps_final, sigma=particle_a.lj_params[particle_b.name][0]
                    )
                    lj.r_cut[(particle_a.name, particle_b.name)] = 1.1  # nm
                else:
                    lj.params[(particle_a.name, particle_b.name)] = dict(
                        epsilon=particle_a.lj_params[particle_b.name][1],
                        sigma=particle_a.lj_params[particle_b.name][0],
                    )
                    lj.r_cut[(particle_a.name, particle_b.name)] = 1.1  # nm
        return lj

    def init_harmonic_bonds(contents, name):
        """
        initialize the bonded potentials of all bonds present in simulation
        according to the Martini3 specifications. Also writes identities of bonds 
        to bonds.csv

        Args:
        contents (list of Molecules): List of all molecules in simulation.
        name (string): path to folder where gsd is saved.

        Returns:
        hoomd.md.bond.Harmonic: Contains all bonds present in simulation.
        """
        bond_harmonic = hoomd.md.bond.Harmonic()

        bond_set = set()
        for molecule in contents.contents:
            for bond in molecule.bonds:
                if type(bond_harmonic.params[str(bond.idx)].get("k")) != type(10):
                    bond_harmonic.params[str(bond.idx)] = dict(
                        k=bond.force, r0=bond.spatial
                    )
                    bond_set.add((bond.idx, bond.force, bond.spatial))

        with open(name + "bonds.csv", "w") as file:
            writer = csv.writer(file)
            for bond in bond_set:
                writer.writerow([bond[0], bond[1], bond[2]])
        return bond_harmonic

    def init_angles(contents, name):
        """
        initialize the angled potentials of all angles present in simulation
        according to the Martini3 specifications. Also saves identities of angles
        to angles.csv

        Args:
        contents (list of molecules): List of all molecules in simulation.
        name (string): path to folder where gsd is saved.

        Returns:
        hoomd.md.angle.Harmonic: Contains all angles present in simulation.
        """
        angle_bonding = hoomd.md.angle.CosineSquared()
        angle_set = set()
        for molecule in contents.contents:
            for angle in molecule.angles:
                if type(angle_bonding.params[str(angle.idx)].get("k")) != type(10):
                    angle_bonding.params[str(angle.idx)] = dict(
                        k=angle.force, t0=float(angle.spatial) / 180 * np.pi
                    )
                    angle_set.add(
                        (angle.idx, angle.force, float(angle.spatial) / 180 * np.pi)
                    )
        with open(name + "angles.csv", "w") as file:
            writer = csv.writer(file)
            for angle in angle_set:
                writer.writerow([angle[0], angle[1], angle[2]])
        return angle_bonding

    def ljPair(self,nl):
        '''
        PEO = -C-O-C-  = s
        Benz= -C=C-    = 
        NN  = -N=N-     =
        '''

        #Lennard-Jones pair interactions
        lj = hoomd.md.pair.LJ(nlist=nl,mode='xplor')
        lj.params[(['PEO','Xlink'],['PEO','Xlink'])] =     dict(epsilon=self.ePEOPEO,sigma=self.sPEOPEO)
        lj.params[(['PEO','Xlink'],'Benz')] =    dict(epsilon=self.ePEOBenz,sigma=self.sPEOBenz)
        lj.params[(['PEO','Xlink'],'NN')] =      dict(epsilon=self.ePEONN,sigma=self.sPEONN)
        lj.params[('Benz','Benz')] =   dict(epsilon=self.eBenzBenz,sigma=self.sBenzBenz)
        lj.params[('Benz','NN')] =     dict(epsilon=self.eBenzNN,sigma=self.sBenzNN)
        lj.params[('NN','NN')] =       dict(epsilon=self.eNNNN,sigma=self.sNNNN)
        lj.r_cut[(['PEO','Xlink'],['PEO','Xlink'])] =   3.0*self.sPEOPEO
        lj.r_cut[(['PEO','Xlink'],'Benz')] =  3.0*self.sPEOBenz
        lj.r_cut[(['PEO','Xlink'],'NN')] =    3.0*self.sPEONN
        lj.r_cut[('Benz','Benz')] = 3.0*self.sBenzBenz
        lj.r_cut[('Benz','NN')] =   3.0*self.sBenzNN
        lj.r_cut[('NN','NN')] =     3.0*self.sNNNN
        lj.params[('Endgroup',['PEO','Benz','NN','Endgroup','Xlink'])] =     dict(epsilon=self.ePEOPEO,sigma=self.sPEOPEO)
        lj.r_cut[('Endgroup',['PEO','Benz','NN','Endgroup','Xlink'])] =     3.0*self.sNNNN
        return lj

    def doubleWellBonds(self):
        v = 3.0
        b = 0.2
        c = 0.5
        a = 2*1.131+2*b
        # r = a/2.0+b
        # Notes: a/2 +- b gives location of both minima, req of B-B(1.131),and 1.131+0.4
        # a/2 gives the location of the energy barrier
        double_well = azplugins.bond.double_well()
        double_well.bond_coeff.set(['AA','AB','BB','BBharm','Xlink','ABharm','AAharm'], V_max=0.0, a=0.0, b=0.5,c=0.0)
        double_well.bond_coeff.set(['sp'], V_max=0.0, a=0.0, b=0.5,c=0.0)
        double_well.bond_coeff.set(['SP','BB_DW'], V_max=v, a=a, b=b,c=c)
        double_well.bond_coeff.set('AB_DW', V_max=v, a=2*1.0465+2*b, b=b,c=c)
        double_well.bond_coeff.set('AA_DW', V_max=v, a=2*0.96+2*b, b=b,c=c)
        return double_well

    def harmonicBonds(self):
        #harmonic bonds
        kharm = 7000/3.4
        harmonic = hoomd.md.bond.Harmonic()
        harmonic.params['PEOPEO'] =     dict(k=kharm, r0=self.sPEOPEO)
        harmonic.params['BenzPEO'] =    dict(k=kharm, r0=self.sPEOBenz)
        harmonic.params['NNPEO'] =      dict(k=kharm, r0=self.sPEONN)
        harmonic.params['BenzBenz'] =   dict(k=kharm, r0=self.sBenzBenz)
        harmonic.params['BenzNN'] =     dict(k=kharm, r0=self.sBenzNN)
        harmonic.params['NNNN'] =       dict(k=kharm, r0=self.sNNNN)
        harmonic.params['EndgroupPEO'] =       dict(k=kharm, r0=self.sNNNN)
        harmonic.params['BenzEndgroup'] =       dict(k=kharm, r0=self.sNNNN)
        harmonic.params['EndgroupNN'] =       dict(k=kharm, r0=self.sNNNN)
        harmonic.params['EndgroupEndgroup'] =       dict(k=kharm, r0=self.sNNNN)
        harmonic.params['PEOXlink'] =       dict(k=kharm, r0=self.sPEOPEO)
        harmonic.params['BenzXlink'] =   dict(k=kharm, r0=self.sPEOBenz)
        harmonic.params['NNXlink'] =      dict(k=kharm, r0=self.sPEONN)
        harmonic.params['EndgroupXlink'] =     dict(k=kharm, r0=self.sPEOEndgroup)
        harmonic.params['XlinkXlink'] =       dict(k=kharm, r0=self.sPEOPEO)
        harmonic.params['Xlink'] =          dict(k=kharm,r0=self.sPEOPEO)
        return harmonic

    def angles(self):
        #angles
        anglepot = hoomd.md.angle.CosineSquared()
        anglepot.params['transAzo'] = dict(k=100.0,t0=math.pi)
        anglepot.params['cisAzo'] = dict(k=100.0, t0=2*math.pi/6)
        anglepot.params['polymer'] = dict(k=400/3.4, t0=2*math.pi*122/360)
        return anglepot

    def crosslinkBonds(self,x_V_barrier,D):
        #----------------Crosslink potential--------------
        xlinkLJ = azplugins.bond.xlink()
        xlinkLJ.bond_coeff.set(['AA','AB','BB','BB_DW','AB_DW','AA_DW','BBharm','ABharm','AAharm'],k4=0.0,r0=1.5,sigma=0,epsilon=0,lj_delta=0,b1=0,b2=0,v0=0)
        xlinkLJ.bond_coeff.set(['SP','sp'],k4=0.0,r0=1.5,sigma=0,epsilon=0,lj_delta=0,b1=0,b2=0,v0=0)
        req = 2**(1/6)*self.sAA
        r0 = 1.3
        b1 = 0
        b2 = 4/3*(req-r0)
        k4 = -1*x_V_barrier/((req-r0-b1)*(req-r0-b2)*(req-r0)**2)
        xlinkLJ.bond_coeff.set('Xlink',k4=k4,r0=r0,sigma=self.sAA-D/(2**(1/6)),epsilon=self.eAA,lj_delta=D,b1=0,b2=b2,v0=x_V_barrier)
        return xlinkLJ

    def backboneBonds(self,V_barrier):
        if (V_barrier == 'FENE'):
            fene = hoomd.md.bond.fene()
            fene.bond_coeff.set('AA', k=30.0, r0=1.5*self.sAA, sigma=self.sAA, epsilon = self.eAA)
            fene.bond_coeff.set('AB', k=30.0, r0=1.5*self.sAB, sigma=self.sAB, epsilon = self.eAB)
            fene.bond_coeff.set('BB', k=30.0, r0=1.5*self.sBB, sigma=self.sBB, epsilon = self.eBB)
            fene.bond_coeff.set(['AA_DW','AAharm','AB_DW','ABharm','BB_DW','BBharm','Xlink'], k=0.0,  r0=1.5*self.sBB, sigma=0, epsilon = 0)
            fene.bond_coeff.set(['SP','sp'], k=0.0,  r0=1.5*self.sBB, sigma=0, epsilon = 0)
            return fene
        else:
            try:
                V_barrier = float(V_barrier)
            except ValueError:
                print('ERROR: Invalid parameter passed to -b')
                exit(2)
            from scipy.optimize import curve_fit

            #Default quartic parameters from literature. (Tsige and Stevens)
            b1 = -0.7589
            b2 = 0
            V0 = 67.2234

            #--------Quartic covalent bond interactions-------------
            x = np.linspace(0.75,2.0,num=100000)

            quartic = azplugins.bond.quartic()
            req = 0.96
            V_min = 20.20
            V_fene = FENE_potential(x,1.0,1.0,1.5,30)
            xdata = x[(V_fene < V_min+V_barrier/4)]
            ydata = V_fene[(V_fene < V_min+V_barrier/4)]
            popt, pcov = curve_fit(quartic_bond, xdata, ydata,maxfev=10000,\
                                p0=[ 1434.3, 0, V_min+V_barrier,        1,      1,     1.5],\
                            bounds=[(-2000,   -0.000001, V_min+V_barrier-0.0001, 0.99999,0.9999,1.499999),
                                    (2000,   0.000001, V_min+V_barrier+0.0001, 1.00001,1.0001,1.500001)])
            b2 = (-30*req/(1-req**2/popt[5]**2)-3*popt[1]*popt[0]*(req-popt[5])**2+4*popt[0]
                *(req-popt[5])**3)/(-2*popt[1]*popt[0]*(req-popt[5])+3*popt[0]*(req-popt[5])**2)
            quartic.bond_coeff.set('AA',k=popt[0],r0=popt[5],sigma=popt[3],epsilon=popt[4],delta=0,b1=popt[1],b2=b2,V0=popt[2])
            # quartic.bond_coeff.set('AA',k=126.208,r0=1.5,sigma=sAA,epsilon=1.0,delta=0,b1=b1,b2=b2,V0=30.2)

            V_min = 23.896
            V_fene = FENE_potential(x,1.1,1.0,1.5*1.1,30)
            xdata = x[(V_fene < V_min+V_barrier/4)]
            ydata = V_fene[(V_fene < V_min+V_barrier/4)]
            popt, pcov = curve_fit(quartic_bond, xdata, ydata,maxfev=10000,\
                                p0=[ 1434.3, 0, V_min+V_barrier,        1*1.1,      1,     1.5*1.1],\
                            bounds=[(-2000,   -0.000001, V_min+V_barrier-0.0001, 0.99999*1.1,0.9999,1.499999*1.1),
                                    (2000,   0.000001, V_min+V_barrier+0.0001, 1.00001*1.1,1.0001,1.500001*1.1)])
            req = 1.0465
            b2 = (-30*req/(1-req**2/popt[5]**2)-3*popt[1]*popt[0]*(req-popt[5])**2+4*popt[0]
                *(req-popt[5])**3)/(-2*popt[1]*popt[0]*(req-popt[5])+3*popt[0]*(req-popt[5])**2)
            quartic.bond_coeff.set('AB',k=popt[0],r0=popt[5],sigma=popt[3],epsilon=popt[4],delta=0,b1=popt[1],b2=b2,V0=popt[2])
            # quartic.bond_coeff.set('AB',k=50.533,r0=1.5,sigma=sAB,epsilon=1.0,delta=0,b1=b1,b2=b2,V0=33.896)

            V_min = 27.842
            V_fene = FENE_potential(x,1.2,1.0,1.5*1.2,30)
            xdata = x[(V_fene < V_min+V_barrier/4)]
            ydata = V_fene[(V_fene < V_min+V_barrier/4)]
            popt, pcov = curve_fit(quartic_bond, xdata, ydata,maxfev=10000,\
                                p0=[ 1434.3, 0, V_min+V_barrier,        1*1.2,      1,     1.5*1.2],\
                            bounds=[(-2000,   -0.000001, V_min+V_barrier-0.0001, 0.99999*1.2,0.9999,1.499999*1.2),
                                    (2000,   0.000001, V_min+V_barrier+0.0001, 1.00001*1.2,1.0001,1.500001*1.2)])
            req = 1.131
            b2 = (-30*req/(1-req**2/popt[5]**2)-3*popt[1]*popt[0]*(req-popt[5])**2+4*popt[0]
                *(req-popt[5])**3)/(-2*popt[1]*popt[0]*(req-popt[5])+3*popt[0]*(req-popt[5])**2)
            quartic.bond_coeff.set('BB',k=popt[0],r0=popt[5],sigma=popt[3],epsilon=popt[4],delta=0,b1=popt[1],b2=b2,V0=popt[2])
            # quartic.bond_coeff.set('BB',k=12.506,r0=1.5,sigma=sBB,epsilon=1.0,delta=0,b1=b1,b2=b2,V0=37.842)

            # quartic.bond_coeff.set(['DW','BS'], k=0.0,  r0=10*sBB, sigma=0, epsilon = 0, delta=0,b1=-0.7589,b2=0,V0=67.2234)
            quartic.bond_coeff.set(['Xlink','BB_DW','AB_DW','AA_DW','BBharm','ABharm','AAharm'],k=0.0,r0=1.5,sigma=0,epsilon=0,delta=0,b1=b1,b2=b2,V0=V0)
            quartic.bond_coeff.set(['SP','sp'],k=0.0,r0=1.5,sigma=0,epsilon=0,delta=0,b1=b1,b2=b2,V0=V0)
            return quartic


