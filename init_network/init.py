from make_init import RandomNetworkMaker
import numpy as np
import sys
sys.path.insert(0,'/Users/statt/Programs/hoomd-4.8.2/')
import hoomd


num_backbone=20
length_backbone=100

crosslinking_probability=0.3 # 30% of all backbone monomers have crosslinkers
azo_probability = 0.5 # 50% of all side chains (non crosslinkers) are ENIM-thethered azo

length_crosslinkers=13
length_sidechains=9
# guess for dense melt synthesis/crosslinking conditions
rho_target= 0.8

# probably should be an even number
N_ionic_liquid = 10000

Ntotal_estimated = (length_backbone*num_backbone)*(crosslinking_probability*length_crosslinkers +\
                                                        (1-crosslinking_probability)*length_sidechains +\
                                                        1) + N_ionic_liquid

L = (Ntotal_estimated/rho_target)**(1/3)

init = RandomNetworkMaker(L,
                          num_backbone,
                          length_backbone,
                          crosslinking_probability,
                          length_crosslinkers,
                          length_sidechains,
                          azo_probability,
                          N_ionic_liquid)


init.create_system()
init.save_system('init.gsd')

cpu = hoomd.device.CPU()

sim = hoomd.Simulation(device=cpu, seed=568)
sim.create_state_from_gsd(filename='init.gsd')

particle_types = sim.state.particle_types
bond_types = sim.state.bond_types


cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell,default_r_cut=2**(1/6.))
lj.params[particle_types,particle_types] = dict(epsilon=1,sigma=1)
lj.mode = 'shift'

gaussian = hoomd.md.pair.DPDConservative(nlist=cell,default_r_cut=1.2)
gaussian.params[particle_types,particle_types]= dict(A=1)
gaussian.mode = 'none'

harmonic = hoomd.md.bond.Harmonic()
harmonic.params[bond_types] = dict(k=1, r0=0.96)

displacement_capped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(),
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
    harmonic.params[bond_types] = dict(k=Q, r0=0.96)
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
                                        couple="none")
integrator.methods = [npt]
integrator.forces = [lj,harmonic]
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(),kT=1.0)
sim.run(10_000)
