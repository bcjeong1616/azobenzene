from scripts.make_init import RandomNetworkMaker
import os
home_directory = os.path.expanduser("~")
import csv
import numpy as np
import sys
sys.path.insert(0,'/Users/statt/Programs/hoomd-4.8.2/')
sys.path.insert(0,home_directory + "/updateAzplugins/hoomd-4.8.2-install/")
sys.path.insert(0,home_directory + "/projects/updateAzplugins/hoomd-4.8.2-install/")
import hoomd
from hoomd import azplugins
print("hoomd: ", hoomd.__file__)
print("azplugins: ",azplugins.__file__)

class Simulator():
    def __init__(self,job):
        
        self.job = job
        # initialization statepoints
        self.num_backbone = job.sp['num_backbone']
        # equilibration statepoints
        self.kT = job.sp['kT']
        self.P = job.sp['P']
        #system statepoints
        self.length_backbone = job.sp['length_backbone']
        self.chain_frac_Azo = job.sp['chain_frac_Azo']  
        self.azo_architecture = job.sp['azo_architecture']      
        self.azo_isomer = job.sp['azo_isomer']
        self.isomerization_scheme = job.sp['isomerization_scheme']
        self.mol_frac_IL = job.sp['mol_frac_IL']
        self.xlink_frac = job.sp['xlink_frac']

        # file names
        self.minimize_gsd_file = job.fn('minimize.gsd')
        self.init_gsd_file = job.fn('init.gsd')
        self.isomerize_gsd_file = job.fn('isomerize.gsd')
        self.exist_gsd_file = job.fn('exist.gsd')

    def initialize(self):
        azo_architecture = self.azo_architecture

        num_backbone=self.num_backbone
        length_backbone=self.length_backbone

        crosslinking_probability = self.xlink_frac # 5% of all side chains are crosslinkers
        azo_probability = self.chain_frac_Azo # 50% of all side chains (non crosslinkers) are EMIM-tethered azo

        length_crosslinkers=13
        length_sidechains=9
        # guess for dense melt synthesis/crosslinking conditions
        rho_target= 0.6#0.8

        # probably should be an even number
        N_ionic_liquid = 2*azo_probability*num_backbone*length_backbone # 10000

        Ntotal_estimated = (length_backbone*num_backbone)*(crosslinking_probability*(length_crosslinkers+4) +\
                (azo_probability*(17+1)) +\
                (1-crosslinking_probability-azo_probability)*(length_sidechains+2)) +\
                N_ionic_liquid*4

        L = (Ntotal_estimated/rho_target)**(1/3)

        init = RandomNetworkMaker(L,
                                num_backbone,
                                length_backbone,
                                crosslinking_probability,
                                length_crosslinkers,
                                length_sidechains,
                                azo_probability,
                                N_ionic_liquid,
                                azo_architecture=azo_architecture,
                                azo_isomer=self.azo_isomer,)


        init.create_system()
        init.save_system(self.minimize_gsd_file)

        cpu = hoomd.device.CPU()

        sim = hoomd.Simulation(device=cpu, seed=568)
        sim.create_state_from_gsd(filename=self.minimize_gsd_file)

        particle_types = sim.state.particle_types
        bond_types = sim.state.bond_types


        cell = hoomd.md.nlist.Cell(buffer=0.4)

        lj = hoomd.md.pair.LJ(nlist=cell,default_r_cut=2**(1/6.)*0.49)#2.5*0.35)
        lj.params[particle_types,particle_types] = dict(epsilon=1,sigma=0.35)
        lj.mode = 'shift'

        gaussian = hoomd.md.pair.DPDConservative(nlist=cell,default_r_cut=1.2)
        gaussian.params[particle_types,particle_types]= dict(A=1)
        gaussian.mode = 'none'

        harmonic_bond = hoomd.md.bond.Harmonic()
        harmonic_bond.params[bond_types] = dict(k=1, r0=0.3)

        displacement_capped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(),
                                                                        maximum_displacement=0.1)
                                                                        # maximum_displacement=0.01)

        fire = hoomd.md.minimize.FIRE(dt=0.05,
                                    force_tol=1e-4,
                                    angmom_tol=1e-2,
                                    energy_tol=1e-4)

        fire.methods = [displacement_capped]
        sim.operations.integrator = fire

        gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(10_000),
                            mode='wb',
                            filename=self.minimize_gsd_file)
        sim.operations.writers.append(gsd)
        
        #-------------------------------------------------------
        #               First Minimization
        #-------------------------------------------------------
        print("Minimize FIRE gaussian,harmonic_bond")
        fire.forces = [harmonic_bond,gaussian]

        gaussian.params[particle_types,particle_types]= dict(A=0.1)
        harmonic_bond.params[bond_types] = dict(k=1, r0=0.3)
        while not fire.converged:
            print("FIRE looped")
            sim.run(100)

        fire.forces = []
        fire.methods = []
        del fire

        #-------------------------------------------------------
        #               Second Minimization with Martini
        #-------------------------------------------------------
        print("Minimize FIRE Martini lj, harmonic_bond")
        lj,harmonic_bond, harmonic_angle, cosine_sq, azo_angle, dihedral, table = init_forces(sim,cell)
        martini_forces = [lj,harmonic_bond,harmonic_angle,cosine_sq,azo_angle,dihedral,table]
        displacement_capped = hoomd.md.methods.DisplacementCapped(
                                            filter=hoomd.filter.All(),
                                            maximum_displacement=0.000001)
        fire = hoomd.md.minimize.FIRE(dt=0.05,
                                    force_tol=1e-2,
                                    angmom_tol=1e-2,
                                    energy_tol=1e-4)
        fire.methods = [displacement_capped]
        fire.forces = martini_forces
        sim.operations.integrator = fire
        # while not fire.converged:
        #     print("Fire LJ exec")
        #     sim.run(100)
        sim.run(10_000)

        fire.forces = []

        #-------------------------------------------------------
        #       NPT relaxation with Martini and dt ramp
        #-------------------------------------------------------
        # NPT relaxation
        print("NPT dt = 0.001 start")
        integrator = hoomd.md.Integrator(dt=0.001)
        sim.operations.integrator = integrator
        npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),
                                                tauS=12.0,
                                                # tauS=1.0,
                                                S=1.0,
                                                # thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.0),
                                                thermostat=hoomd.md.methods.thermostats.Bussi(kT=2.58,tau=1.0),
                                                # thermostat=hoomd.md.methods.thermostats.MTTK(kT=1.0),
                                                couple="xyz")
        integrator.methods = [npt]
        integrator.forces = martini_forces
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(),kT=1.0)
        sim.run(1_000)

        print("NPT dt = 0.005")
        integrator.dt = 0.005
        sim.run(1_000)

        print("NPT dt = 0.02")
        integrator.dt = 0.02
        sim.run(1_000)

        # Make sure that the last frame is printed out
        gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1),
                            mode='wb',
                            filename=self.init_gsd_file)
        sim.operations.writers.append(gsd)
        sim.run(1)

    def isomerize(self):
        try:
            device = hoomd.device.GPU()
            print("GPU detected")
        except:
            device = hoomd.device.CPU()
            print("CPU detected")

        sim = hoomd.Simulation(device=device, seed=568)

        cell = hoomd.md.nlist.Cell(buffer=0.4)

        #-------------------------------------------------------
        #               Change the azo isomer
        #-------------------------------------------------------
        import gsd.hoomd
        traj = gsd.hoomd.open(name=self.init_gsd_file,mode='r')
        frame = traj[-1]

        angle_types = frame.angles.types
        cis_id = angle_types.index('azo_cis_isomer')
        trans_id = angle_types.index('azo_trans_isomer')
        angle_ids = frame.angles.typeid
        azo_angles = np.where(angle_ids == trans_id)[0]
        angle_ids[azo_angles] = cis_id
        frame.angles.typeid = angle_ids

        sim.create_state_from_snapshot(frame)

        #-------------------------------------------------------
        #               Set up writers
        #-------------------------------------------------------

        gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1_000),
                            mode='wb',
                            filename=self.isomerize_gsd_file)
        sim.operations.writers.append(gsd)

        thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        sim.operations.computes.append(thermodynamic_properties)

        logger = hoomd.logging.Logger(categories=['scalar'])
        logger.add(sim, quantities=['timestep','tps'])
        logger.add(thermodynamic_properties, quantities=['kinetic_temperature','pressure','kinetic_energy','potential_energy','volume'])
        table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),
                                    logger=logger,
                                    output=open(self.job.fn('isomerize.log'),'w'))
        sim.operations.writers.append(table_file)

        table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger)
        sim.operations.writers.append(table_stdout)

        #-------------------------------------------------------
        #               Set up integrator and Run
        #-------------------------------------------------------
        lj,harmonic_bond, harmonic_angle, cosine_sq, azo_angle,dihedral, table= init_forces(sim,cell)
        martini_forces = [lj,harmonic_bond,harmonic_angle,cosine_sq,azo_angle,dihedral,table]
        
        integrator = hoomd.md.Integrator(dt=0.02)
        sim.operations.integrator = integrator
        npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),
                                                tauS=12.0,
                                                S=1.0,
                                                # thermostat=hoomd.md.methods.thermostats.Bussi(kT=2.58,tau=1.0),
                                                thermostat=hoomd.md.methods.thermostats.MTTK(kT=2.58,tau=1.0),
                                                couple="xyz")
        integrator.methods = [npt]
        integrator.forces = martini_forces
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(),kT=2.58)
        sim.run(10_000)

    def exist(self):
        try:
            device = hoomd.device.GPU()
            print("GPU detected")
        except:
            device = hoomd.device.CPU()
            print("CPU detected")

        sim = hoomd.Simulation(device=device, seed=568)
        if self.azo_isomer == 'trans_to_cis':
            sim.create_state_from_gsd(filename=self.isomerize_gsd_file)
        else:
            sim.create_state_from_gsd(filename=self.init_gsd_file)

        cell = hoomd.md.nlist.Cell(buffer=0.4)

        #-------------------------------------------------------
        #               Set up writers
        #-------------------------------------------------------

        gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1_000),
                            mode='wb',
                            filename=self.exist_gsd_file)
        sim.operations.writers.append(gsd)

        thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        sim.operations.computes.append(thermodynamic_properties)

        logger = hoomd.logging.Logger(categories=['scalar'])
        logger.add(sim, quantities=['timestep','tps'])
        logger.add(thermodynamic_properties, quantities=['kinetic_temperature','pressure','kinetic_energy','potential_energy','volume'])
        table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),
                                    logger=logger,
                                    output=open(self.job.fn('exist.log'),'w'))
        sim.operations.writers.append(table_file)

        table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger)
        sim.operations.writers.append(table_stdout)

        #-------------------------------------------------------
        #               Set up integrator and Run
        #-------------------------------------------------------
        lj,harmonic_bond, harmonic_angle, cosine_sq,azo_angle, dihedral, table= init_forces(sim,cell)
        martini_forces = [lj,harmonic_bond,harmonic_angle,cosine_sq,azo_angle,dihedral,table]
        
        integrator = hoomd.md.Integrator(dt=0.02)
        sim.operations.integrator = integrator
        npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),
                                                tauS=12.0,
                                                S=1.0,
                                                # thermostat=hoomd.md.methods.thermostats.Bussi(kT=2.58,tau=1.0),
                                                thermostat=hoomd.md.methods.thermostats.MTTK(kT=2.58,tau=1.0),
                                                couple="xyz")
        integrator.methods = [npt]
        integrator.forces = martini_forces
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(),kT=2.58)
        sim.run(100_000)

#--------------------------------------------------
#                   Force field
#--------------------------------------------------
def init_forces(sim,cell):
    # self.particle_types = ['TC1','SN4a','SN3r','SC1','TN3r','TC5','TN2q','TC6','TC3','SX4e','SQ4n',]
    # self.bond_types = ['TC1-TC1','TC1-SN4a','SN4a-SN3r','SN3r-SN3r',
    #                     'SN4a-SC1','SC1-SC1','SC1-TN3r','TN3r-TC5',
    #                     'TC5-TC5','SC1-TN2q',
    #                     'SX4e-SQ4n','SQ4n-SQ4n','TN2q-TN2q','TN2q-TC3','TN2q-TC6']
    # self.angle_types = ['generic','backbone','PEO','azo_ring','azo_trans_isomer','azo_cis_isomer','TFSI','EMIM']
    # self.dihedral_types = ['PEO','TFSI']
    particle_types = sim.state.particle_types
    bond_types = sim.state.bond_types
    angle_types = sim.state.angle_types
    dihedral_types = sim.state.dihedral_types

    lj = hoomd.md.pair.LJ(nlist=cell)
    for i in range(len(particle_types)):
        for j in range(i,len(particle_types)):
            particle_a = particle_types[i]
            particle_b = particle_types[j]
            sigma, eps = lj_lookup(particle_a,particle_b)
            lj.params[(particle_a, particle_b)] = dict(epsilon=eps,sigma=sigma)
            lj.r_cut[(particle_a, particle_b)] = 1.1

    harmonic_bond = hoomd.md.bond.Harmonic()
    constraint_k = 30000#7000
    default_k = 7000
    harmonic_bond.params['TC1-TC1'] =    dict(k=default_k, r0=0.25)
    harmonic_bond.params['TC1-SN4a'] =   dict(k=default_k, r0=0.30)
    harmonic_bond.params['SN4a-SN3r'] =  dict(k=default_k, r0=0.35)
    harmonic_bond.params['SN3r-SN3r'] =  dict(k=7000, r0=0.36) # PEO model Polyply
    harmonic_bond.params['SN4a-SC1'] =   dict(k=default_k, r0=0.35)
    harmonic_bond.params['SC1-SC1'] =    dict(k=default_k, r0=0.35)
    harmonic_bond.params['SC1-TN3r'] =   dict(k=default_k, r0=0.30)
    harmonic_bond.params['TN3r-TC5'] =   dict(k=default_k, r0=0.25)
    harmonic_bond.params['TC5-TC5'] =    dict(k=constraint_k, r0=0.29) # constraint_k:Zheng et al, 0.29:Souza et al CoG
    harmonic_bond.params['SC1-TN2q'] =   dict(k=default_k, r0=0.30)
    harmonic_bond.params['TN2q-TN2q'] =  dict(k=constraint_k, r0=0.318) # Barbosa et al 2022
    harmonic_bond.params['TN2q-TC6'] =   dict(k=constraint_k, r0=0.318) # Barbosa et al 2022
    harmonic_bond.params['TN2q-TC3'] =   dict(k=default_k, r0=0.25)
    harmonic_bond.params['SX4e-SQ4n'] =  dict(k=constraint_k, r0=0.277) # Grunewald 2018 Thesis
    harmonic_bond.params['SQ4n-SQ4n'] =  dict(k=constraint_k, r0=0.247) # Grunewald 2018 Thesis
    # bond_constraints = hoomd.md.Constrain.Distance()
    # #constraint distances are given by the system state, so make sure FIRE succeeds first
    # bond_constraints.constraint_group

    angle_type_list = ['generic','backbone','PEO','azo_ring','azo_trans_isomer','azo_cis_isomer','TFSI1','TFSI2','EMIM']
    generic_k = 30
    harmonic_angle = hoomd.md.angle.Harmonic()
    harmonic_angle.params['generic'] = dict(k=0.0001,t0=np.pi) # None, should have no influence. Uses cosine sq
    harmonic_angle.params['backbone'] = dict(k=78,t0=119/180*np.pi) # PP model from Panizon 2015
    harmonic_angle.params['PEO'] = dict(k=80,t0=123/180*np.pi) # polyply PEO
    harmonic_angle.params['azo_ring'] = dict(k=25,t0=np.pi) # Zheng et al 
    # harmonic_angle.params['azo_trans_isomer'] = dict(k=30,t0=np.pi) # Zheng et al
    # harmonic_angle.params['azo_cis_isomer'] = dict(k=30,t0=60/180*np.pi) # Li et al 2014 for t0
    harmonic_angle.params['azo_trans_isomer'] = dict(k=0.0001,t0=np.pi) # Zheng et al
    harmonic_angle.params['azo_cis_isomer'] = dict(k=0.0001,t0=60/180*np.pi) # Li et al 2014 for t0
    harmonic_angle.params['TFSI1'] = dict(k=generic_k,t0=125/180*np.pi) # Grunewald 2018
    harmonic_angle.params['TFSI2'] = dict(k=generic_k,t0=90/180*np.pi) # Grunewald 2018
    harmonic_angle.params['EMIM'] = dict(k=50,t0=120/180*np.pi) # Barbosa 2022
    cosine_sq = hoomd.md.angle.CosineSquared()
    cosine_sq.params['generic'] = dict(k=50,t0=110/180*np.pi) # Paul et al 1995
    downselected_list = angle_type_list.copy()
    downselected_list.remove('generic')
    cosine_sq.params[downselected_list] = dict(k=0.0001,t0=np.pi)
    def li_et_al(t0,k,C,D, table_width):
        theta = np.linspace(0,np.pi,table_width)
        U = 0.5*k*(theta-t0)**2*(1+C*(theta-t0)+D*(theta-t0)**2)
        torque =  -k/2*2*(theta-t0)+k/2*3*C*(theta-t0)**2+k/2*4*D*(theta-t0)**3
        return U,torque
    width = 1000
    trans_azo_FF_U, trans_azo_FF_T = li_et_al(t0=180/180*np.pi,k=105*4.184,
                            C=0.5,D=0.073,
                            table_width=width)
    cis_azo_FF_U, cis_azo_FF_T = li_et_al(t0=60/180*np.pi,k=98*4.184,
                          C=-0.532,D=0.08,
                          table_width=width)
    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0,np.pi,width),trans_azo_FF[0])
    # plt.plot(np.linspace(0,np.pi,width),cis_azo_FF[0])
    # plt.ylim(-1,80*5)
    # plt.show()
    # exit()
    azo_angle = hoomd.md.angle.Table(width=width)
    azo_angle.params['azo_trans_isomer'] = dict(U=trans_azo_FF_U,tau=trans_azo_FF_T)
    azo_angle.params['azo_cis_isomer'] = dict(U=cis_azo_FF_U,tau=cis_azo_FF_T)
    downselected_list = angle_type_list.copy()
    downselected_list.remove('azo_trans_isomer')
    downselected_list.remove('azo_cis_isomer')
    azo_angle.params[downselected_list] = dict(U=np.zeros(width),tau=np.zeros(width))

    dihedral = hoomd.azplugins.dihedral.BendingTorsion()
    dihedral.params['PEO'] = dict(k_phi=0.6570,a0=-1.3278,a1=-0.43661278,a2=1.0808704,
                                a3=0.680055,a4=0.0) # Polyply
    dihedral.params['TFSI'] = dict(k_phi=0,a0=0,a1=0,a2=0,a3=0,a4=0) #None. Just going to assume that it isn't significant

    # improper = hoomd.md.dihedral.Improper()
    # improper.params['PEO'] = dict()
    # improper.params['TFSI'] = dict()

    # Coulomb interactions
    def coulomb_interaction(r_min, r_max, q1, q2, eps_r, eps_0, table_width):
        r = np.linspace(r_min,r_max, table_width)
        U = 1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r) - 1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r_max)
        F =  1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r**2)
        return U,F
    table = hoomd.md.pair.Table(nlist=cell, default_r_cut=0)
    table.params[(particle_types,particle_types)] = dict(r_min=0, U=[0], F=[0])
    table.r_cut[(particle_types,particle_types)] = 0
    import scipy.constants as const
    import math
    e = const.e
    p = const.epsilon_0
    SI_kT = const.k * 2.5/0.0083144626181532
    eps_0 = p*1e-9*SI_kT/(e**2)                # permittivity in simulation units from ________
    eps_r = 7 # starting here at 7, based on eps_r for DXE according to 
    bjerrum_length = 1 / (4 * math.pi * eps_0 * eps_r)
    print(bjerrum_length)
    r_max = 1.1
    # r_max = bjerrum_length
    TQ5_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=0.75, q2=0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    SP1q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=-0.5*0.75, q2=-0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    TN2q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=0.5*0.75, q2=0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    TQ5xSP1q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=0.75, q2=-0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    TQ5xTN2q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=0.75, q2=0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    SP1qxTN2q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=-0.5*0.75, q2=0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
    table.params[('TQ5','TQ5')] = dict(r_min=0.1,   U=TQ5_coulomb_interaction[0],       F=TQ5_coulomb_interaction[1])
    table.r_cut[('TQ5','TQ5')] = r_max
    table.params[('SP1q','SP1q')] = dict(r_min=0.1, U=SP1q_coulomb_interaction[0],      F=SP1q_coulomb_interaction[1])
    table.r_cut[('SP1q','SP1q')] = r_max
    table.params[('TN2q','TN2q')] = dict(r_min=0.1, U=TN2q_coulomb_interaction[0],      F=TN2q_coulomb_interaction[1])
    table.r_cut[('TN2q','TN2q')] = r_max
    table.params[('TQ5','SP1q')] = dict(r_min=0.1,  U=TQ5xSP1q_coulomb_interaction[0],  F=TQ5xSP1q_coulomb_interaction[1])
    table.r_cut[('TQ5','SP1q')] = r_max
    table.params[('TQ5','TN2q')] = dict(r_min=0.1,  U=TQ5xTN2q_coulomb_interaction[0],  F=TQ5xTN2q_coulomb_interaction[1])
    table.r_cut[('TQ5','TN2q')] = r_max
    table.params[('SP1q','TN2q')] = dict(r_min=0.1, U=SP1qxTN2q_coulomb_interaction[0], F=SP1qxTN2q_coulomb_interaction[1])
    table.r_cut[('SP1q','TN2q')] = r_max

    return lj,harmonic_bond, harmonic_angle, cosine_sq, azo_angle,dihedral, table

def nth_particle(row_text, item_number):
    """
    Splits the text in a row of text into an array, finding the nth piece of information from that row

    Args:
      row_text: One row of text document (in this case the itp)
      item_number (int): if mass, item number is one. Otherwise it is whatever.

    Returns:
      item (str): nth piece of text present in row
    """
    iter = 0
    for item in row_text:
        if item != "" and iter == item_number:
            return item
        elif item != "":
            iter = iter + 1

def lj_lookup(particle_name_A, particle_name_B):
    """
    For a given particle, a dict of all the other particle names (key) and their lj params (sigma,epsilon) with the given particle

    Args:
      particle name (string)
      particle names (list of strings)

    Returns:
      lj_potentials (dict): dict of all particle names and the tuple of LJ potentials
    """
    # root = os.path.abspath(os.sep)
    home = os.path.expanduser("~")
    forces = home + "/programs" + "/martini3/" + "martini_v3.0.0.itp"
    with open(forces, newline="") as f:
        reader = csv.reader(f)
        lj_potentials = {}
        nonbond_params_point = False
        for row in reader:
            if row != []:
                if row[0] == "[ nonbond_params ]":
                    nonbond_params_point = True
                if nonbond_params_point:
                    row_text = row[0].split(" ")
                    p1_name = nth_particle(row_text, 0)
                    p2_name = nth_particle(row_text, 1)
                    if p1_name == particle_name_A:
                        if p2_name == particle_name_B:
                            sigma = float(nth_particle(row_text, 3))
                            eps = float(nth_particle(row_text, 4))
                            return sigma, eps
                    if p2_name == particle_name_A:
                        if p1_name == particle_name_B:
                            sigma = float(nth_particle(row_text, 3))
                            eps = float(nth_particle(row_text, 4))
                            return sigma, eps

# if __name__ == "__main__":
#     main()

