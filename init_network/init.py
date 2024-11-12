from make_init import RandomNetworkMaker
import numpy as np
import sys
sys.path.insert(0,'/Users/statt/Programs/hoomd-4.8.2/')
sys.path.insert(0,"/home/bj21/updateAzplugins/hoomd-4.8.2-install/")
import hoomd
from hoomd import azplugins
print(hoomd.__file__)

azo_architecture = 'sideChainIL'

num_backbone=20#20
length_backbone=100#100

crosslinking_probability=0.0 # 30% of all backbone monomers have crosslinkers
azo_probability = 0.05 # 50% of all side chains (non crosslinkers) are EMIM-tethered azo

length_crosslinkers=13
length_sidechains=9
# guess for dense melt synthesis/crosslinking conditions
rho_target= 0.8

# probably should be an even number
N_ionic_liquid = 2*azo_probability*num_backbone*length_backbone # 10000

Ntotal_estimated = (length_backbone*num_backbone)*(crosslinking_probability*(length_crosslinkers+4) +\
                                                (azo_probability*(17+1)) +\
                                                (1-crosslinking_probability-azo_probability)*(length_sidechains+2)) + N_ionic_liquid*4

L = (Ntotal_estimated/rho_target)**(1/3)

init = RandomNetworkMaker(L,
                          num_backbone,
                          length_backbone,
                          crosslinking_probability,
                          length_crosslinkers,
                          length_sidechains,
                          azo_probability,
                          N_ionic_liquid,
                          azo_architecture=azo_architecture)


init.create_system()
init.save_system('init.gsd')

cpu = hoomd.device.CPU()

sim = hoomd.Simulation(device=cpu, seed=568)
sim.create_state_from_gsd(filename='init.gsd')

particle_types = sim.state.particle_types
bond_types = sim.state.bond_types


cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell,default_r_cut=2**(1/6.)*0.35)
lj.params[particle_types,particle_types] = dict(epsilon=1,sigma=0.35)
lj.mode = 'shift'

gaussian = hoomd.md.pair.DPDConservative(nlist=cell,default_r_cut=1.2)
gaussian.params[particle_types,particle_types]= dict(A=1)
gaussian.mode = 'none'

harmonic = hoomd.md.bond.Harmonic()
harmonic.params[bond_types] = dict(k=1, r0=0.3)

displacement_capped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(),
                                                                # maximum_displacement=0.1)
                                                                maximum_displacement=0.01)

fire = hoomd.md.minimize.FIRE(dt=0.05,
                            force_tol=1e-2,
                            angmom_tol=1e-2,
                            energy_tol=1e-3)


fire.methods = [displacement_capped]
sim.operations.integrator = fire

gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(10_000),
                      mode='wb',
                      filename='minimize.gsd')
sim.operations.writers.append(gsd)

print("Minimize FIRE increase gaussian,harmonic")
fire.forces = [harmonic,gaussian]

for Q in [1,10,100,1000]:
    print("Q=",Q)
    gaussian.params[particle_types,particle_types]= dict(A=Q)
    harmonic.params[bond_types] = dict(k=Q, r0=0.3)
    while not fire.converged:
        sim.run(100)

print("Minimize FIRE increase gaussian")
fire.forces = [gaussian,harmonic]

for Q in [1e3,2e3,5e3,1e4]:
    print("Q=",Q)
    gaussian.params[particle_types,particle_types]= dict(A=Q)
    while not fire.converged:
        sim.run(100)

print("Minimize FIRE lj, harmonic")
fire.forces = [lj,harmonic]
sim.run(10000)

fire.forces = []

# NPT relaxation just because
integrator = hoomd.md.Integrator(dt=0.001)
sim.operations.integrator = integrator
npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),
                                        tauS=1.0,
                                        S=1.0,
                                        thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.0),
                                        couple="xyz")
integrator.methods = [npt]
integrator.forces = [lj,harmonic]
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(),kT=1.0)
sim.run(10_000)

#--------------------------------------------------
#                   Force field
#--------------------------------------------------

# self.particle_types = ['TC1','SN4a','SN3r','SC1','TN3r','TC5','TN2q','TC6','TC3','SX4e','SQ4n',]
# self.bond_types = ['TC1-TC1','TC1-SN4a','SN4a-SN3r','SN3r-SN3r',
#                     'SN4a-SC1','SC1-SC1','SC1-TN3r','TN3r-TC5',
#                     'TC5-TC5','SC1-TN2q',
#                     'SX4e-SQ4n','SQ4n-SQ4n','TN2q-TN2q','TN2q-TC3','TN2q-TC6']
# self.angle_types = ['generic','backbone','PEO','azo_ring','azo_trans_isomer','azo_cis_isomer','TFSI','EMIM']
# self.dihedral_types = ['PEO','TFSI']


harmonic = hoomd.md.bond.Harmonic()
constraint_k = 7000#30000
default_k = 30
harmonic.params['TC1-TC1'] =    dict(k=default_k, r0=0.25)
harmonic.params['TC1-SN4a'] =   dict(k=default_k, r0=0.30)
harmonic.params['SN4a-SN3r'] =  dict(k=default_k, r0=0.35)
harmonic.params['SN3r-SN3r'] =  dict(k=7000, r0=0.36) # PEO model Polyply
harmonic.params['SN4a-SC1'] =   dict(k=default_k, r0=0.35)
harmonic.params['SC1-SC1'] =    dict(k=default_k, r0=0.35)
harmonic.params['SC1-TN3r'] =   dict(k=default_k, r0=0.30)
harmonic.params['TN3r-TC5'] =   dict(k=default_k, r0=0.25)
harmonic.params['TC5-TC5'] =    dict(k=constraint_k, r0=0.29) # constraint_k:Zheng et al, 0.29:Souza et al CoG
harmonic.params['SC1-TN2q'] =   dict(k=default_k, r0=0.30)
harmonic.params['TN2q-TN2q'] =  dict(k=constraint_k, r0=0.318) # Barbosa et al 2022
harmonic.params['TN2q-TC6'] =   dict(k=constraint_k, r0=0.318) # Barbosa et al 2022
harmonic.params['TN2q-TC3'] =   dict(k=default_k, r0=0.25)
harmonic.params['SX4e-SQ4n'] =  dict(k=constraint_k, r0=0.277) # Grunewald 2018 Thesis
harmonic.params['SQ4n-SQ4n'] =  dict(k=constraint_k, r0=0.247) # Grunewald 2018 Thesis
# bond_constraints = hoomd.md.Constrain.Distance()
# #constraint distances are given by the system state, so make sure FIRE succeeds first
# bond_constraints.constraint_group





angle_type_list = ['generic','backbone','PEO','azo_ring','azo_trans_isomer','azo_cis_isomer','TFSI','EMIM']
generic_k = 30
harmonic_angle = hoomd.md.angle.Harmonic()
harmonic_angle.params['generic'] = dict(k=0.0001,t0=np.pi) # None, should have no influence. Uses cosine sq
harmonic_angle.params['backbone'] = dict(k=78,t0=119/180*np.pi) # PP model from Panizon 2015
harmonic_angle.params['PEO'] = dict(k=80,t0=123/180*np.pi) # polyply PEO
harmonic_angle.params['azo_ring'] = dict(k=25,t0=np.pi) # Zheng et al 
harmonic_angle.params['azo_trans_isomer'] = dict(k=30,t0=np.pi) # Zheng et al
harmonic_angle.params['azo_cis_isomer'] = dict(k=30,t0=60/180*np.pi) # Li et al 2014 for t0
harmonic_angle.params['TFSI1'] = dict(k=generic_k,t0=125/180*np.pi) # Grunewald 2018
harmonic_angle.params['TFSI2'] = dict(k=generic_k,t0=90/180*np.pi) # Grunewald 2018
harmonic_angle.params['EMIM'] = dict(k=50,t0=120/180*np.pi) # Barbosa 2022
cosine_sq = hoomd.md.angle.CosineSquared()
cosine_sq.params['generic'] = dict(k=50,t0=110/180*np.pi) # Paul et al 1995
downselected_list = angle_type_list.copy()
downselected_list.remove('generic')
cosine_sq.params[downselected_list] = dict(k=0.0001,t0=np.pi)

dihedral = hoomd.azplugins.BendingTorsion()
dihedral.params['PEO'] = dict(k_phi=0.6570,a0=-1.3278,a1=-0.43661278,a2=1.0808704,
                              a3=0.680055,a4=0.0) # Polyply
dihedral.params['TFSI'] = dict(k_phi=0,a0=0,a1=0,a2=0,a3=0,a4=0) #None. Just going to assume that it isn't significant

# improper = hoomd.md.dihedral.Improper()
# improper.params['PEO'] = dict()
# improper.params['TFSI'] = dict()

