import numpy as np
import sys
sys.path.insert(0,"/home/bj21/updateAzplugins/hoomd-4.8.2-install/")
import hoomd
import hoomd.azplugins
sys.path.append('/home/bj21/programs/')
from martini3 import molecules
from martini3 import init_cell
from martini3 import force_fields
import os
import math
import datetime
import random

def gen_grid(low_bound, up_bound, num_spaces=100):
    grid = []
    x_arr = np.linspace(low_bound, up_bound, num_spaces)
    for x in x_arr:
        for y in x_arr:
            grid.append((x, y))
    return grid

def gen_grid_3D(low_bound, up_bound, num_spaces=100):
    grid = []
    x_arr = np.linspace(low_bound, up_bound, num_spaces)
    for x in x_arr:
        for y in x_arr:
            for z in x_arr:
                grid.append((x, y, z))
    return grid

def wrap_pbc(x,box):
        delta = np.where(x > 0.5 * box, x - box, x)
        delta = np.where(delta < - 0.5 * box, box + delta, delta)
        return delta

class Status:
    def __init__(self, simulation):
        self.simulation = simulation

    @property
    def seconds_remaining(self):
        try:
            return (
                self.simulation.final_timestep - self.simulation.timestep
            ) / self.simulation.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))


def main():

    # contents setup
    contents = molecules.Contents()

    kT = 298.15*0.0083144626181532 # convert T to kT, 25C
    salt_mol_frac_arr = [0.01]#0.005 #[0.4,1.0]#0.0 [0.1,0.2,
    P = 0.0602214076 # 16.6053906648 bar/unit: 1 bar
    total_mol = 5000
    init_density = 0.9
    # azo_mass = 182.226 * 1.66054*10**-24 # convert AMU to grams
    init_volume =  total_mol/init_density
    # init_volume = init_volume * 10**21            # convert cm**3 to nm**3
    x_box_size = init_volume**(1/3)
    z_box_size = init_volume**(1/3)
    # print(x_box_size)

    positions = np.empty((0, 3))
    np.random.seed()
    low_bound = -x_box_size / 2 + .25
    up_bound = x_box_size / 2 - .25
    low_bound_z = -z_box_size / 2 + .5
    up_bound_z = z_box_size/2 - .5

    # z_grid = np.linspace(low_bound_z, up_bound_z, int(z_box_size))
    # print(len(grid_polym))

    # num_salts = salt_mol_frac_arr*total_mol
    for salt_mol_frac in salt_mol_frac_arr:
        num_salts = int(round(salt_mol_frac*total_mol))
        num_G4 = total_mol - num_salts
        grid = gen_grid_3D(low_bound, up_bound, num_spaces=int(x_box_size*1.5))

        #create the initial gsd to run the simulation from
        #first, the salt
        for i in range(num_salts):
            #random position
            new_pos = random.choice(grid)
            #random orientation
            alpha = np.random.rand()*2*math.pi
            beta  = np.random.rand()*2*math.pi
            gamma = np.random.rand()*2*math.pi

            molecule = molecules.make_molecule(
                name="TFSI",
                contents=contents,
                x_shift=new_pos[0],
                y_shift=new_pos[1],
                z_shift=new_pos[2],
                alpha=alpha,
                beta=beta,
                gamma=gamma
                )
            #account for PBC:
            molecule_position = np.array(molecule.positions)
            # print(np.shape(molecule.position))
            for j,position in enumerate(molecule.positions):
                molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
            molecule.positions = molecule_position

            contents.add_molecule(molecule)
            grid.remove(new_pos)
            
            #Add lithium
            #random position
            new_pos = random.choice(grid)
            #random orientation
            alpha = np.random.rand()*2*math.pi
            beta  = np.random.rand()*2*math.pi
            gamma = np.random.rand()*2*math.pi
            
            lithium = molecules.make_molecule(
                name="Li+",
                contents=contents,
                x_shift=new_pos[0],
                y_shift=new_pos[1],
                z_shift=new_pos[2],
                alpha=alpha,
                beta=beta,
                gamma=gamma
                )
            #account for PBC:
            molecule_position = np.array(lithium.positions)
            # print(np.shape(lithium.position))
            for j,position in enumerate(lithium.positions):
                molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
            lithium.positions = molecule_position

            contents.add_molecule(lithium)
            grid.remove(new_pos)

        #add the solvent G4
        for i in range(num_G4):
            #random position
            new_pos = random.choice(grid)
            #random orientation
            alpha = np.random.rand()*2*math.pi
            beta  = np.random.rand()*2*math.pi
            gamma = np.random.rand()*2*math.pi
            
            G4_molecule = molecules.make_molecule(
                name="G4",
                contents=contents,
                x_shift=new_pos[0],
                y_shift=new_pos[1],
                z_shift=new_pos[2],
                alpha=alpha,
                beta=beta,
                gamma=gamma
                )
            #account for PBC:
            molecule_position = np.array(G4_molecule.positions)
            # print(np.shape(G4_molecule.position))
            for j,position in enumerate(G4_molecule.positions):
                molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
            G4_molecule.positions = molecule_position

            contents.add_molecule(G4_molecule)
            grid.remove(new_pos)

        #initialize the gsd
        path = "data/" + "LiTFSI_in_G4" + f"_{salt_mol_frac}"+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = init_cell.init_cell(
            contents, path, box_size=[x_box_size, x_box_size, z_box_size], pair_on=False #False
        )
        # init cell also saves the gsd
        print("Initialization for ",salt_mol_frac, " completed, starting equilibration")

        #run the equilibration on the gsd
        equilibrate(init_path=path,kT=kT,P=P)
        
        # exit()

def equilibrate(init_path,kT,P):
    if not os.path.exists(init_path):
        raise Exception("no data present")

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = force_fields.forces_from_gsd(
        init_path, "init.gsd"
    )

    # print(dihedrals)
    # print(dihedrals.params["0"])
    # print(dihedrals.params["0"].keys())
    # print(dihedrals.params["0"].values())
    # exit(0)

    try:
        sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=16)
        print("Running on the GPU")
    except:
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=16)
        print("Running on the CPU")

    name = "init.gsd"

    sim.create_state_from_gsd(filename=init_path + name)

    status = Status(sim)
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(sim, quantities=["timestep", "tps"])
    logger[("Status", "etr")] = (status, "etr", "string")
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(period=5000), logger=logger
    )
    sim.operations.writers.append(table)

    import gsd
    with gsd.hoomd.open(name=init_path + name, mode="r") as f:
        frame = f[0]
        nl = hoomd.md.nlist.Cell(buffer=0.4)
        dpd = hoomd.md.pair.DPD(nlist=nl, kT=1.5, default_r_cut=0.43)
        dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=10, gamma=4.5)

    #-----------------------------------------------------------------------
    #                               Output
    #-----------------------------------------------------------------------

    gsd_file = init_path + 'traj.gsd'
    log_file = init_path + 'traj.log'

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim, quantities=['timestep'])
    logger.add(thermodynamic_properties, quantities=['kinetic_temperature','pressure','kinetic_energy','potential_energy','volume'])
    # table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger)
    table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger, output=open(log_file,'a'))
    # sim.operations.writers.append(table_stdout)
    sim.operations.writers.append(table_file)

    gsd_writer = hoomd.write.GSD(filename=gsd_file,
                                        trigger=hoomd.trigger.Periodic(1000),
                                        dynamic=['property','momentum','topology','attribute'],
                                        mode='wb')
    sim.operations.writers.append(gsd_writer)

    #-----------------------------------------------------------------------
    #                             Integration
    #-----------------------------------------------------------------------

    # apply FIRE minimization to remove overlaps
    mttk = hoomd.md.methods.thermostats.MTTK(kT = 1.5, tau = 1)
    cv = hoomd.md.methods.ConstantVolume(filter = hoomd.filter.All(),thermostat = mttk)
    # langevin = hoomd.md.methods.Langevin(filter = hoomd.filter.All(),kT=2.47)
    fire = hoomd.md.minimize.FIRE(dt=0.0005,
                        force_tol=1e-2,
                        angmom_tol=1e-2,
                        energy_tol=1e-7,
                        methods=[cv],
                        forces=[dpd, bond_harmonic, angle_forces,dihedrals],)
    
    sim.operations.integrator = fire
    sim.run(1e5)

    print("FIRE 1 done")

    for A in np.linspace(10,400,5):
        print("A: ",A)
        dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=A, gamma=4.5)
        fire.forces = [dpd, bond_harmonic, angle_forces,dihedrals]
        sim.operations.integrator = fire
        sim.run(1e5)

    #Replace DPD with LJ
    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = force_fields.forces_from_gsd(
        init_path, "init.gsd"
    )
    traj = gsd.hoomd.open(init_path + "init.gsd",mode='r')
    frame = traj[-1]
    # Coulomb interactions
    def coulomb_interaction(r_min, r_max, q1, q2, eps_r, eps_0, table_width):
        r = np.linspace(r_min,r_max, table_width)
        U = 1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r) - 1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r_max)
        F =  1./(4.*np.pi*eps_0*eps_r) * (q1*q2/r**2)
        return U,F
    table = hoomd.md.pair.Table(nlist=nl, default_r_cut=0)
    table.params[(frame.particles.types, frame.particles.types)] = dict(r_min=0, U=[0], F=[0])
    table.r_cut[(frame.particles.types, frame.particles.types)] = 0
    # Physical constants
    import scipy.constants as const
    e = const.e
    p = const.epsilon_0
    SI_kT = const.k * kT/0.0083144626181532
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
    table.params[('SP1q','SP1q')] = dict(r_min=0.1, U=SP1q_coulomb_interaction[0],      F=SP1q_coulomb_interaction[1])
    table.params[('TN2q','TN2q')] = dict(r_min=0.1, U=TN2q_coulomb_interaction[0],      F=TN2q_coulomb_interaction[1])
    table.params[('TQ5','SP1q')] = dict(r_min=0.1,  U=TQ5xSP1q_coulomb_interaction[0],  F=TQ5xSP1q_coulomb_interaction[1])
    table.params[('TQ5','TN2q')] = dict(r_min=0.1,  U=TQ5xTN2q_coulomb_interaction[0],  F=TQ5xTN2q_coulomb_interaction[1])
    table.params[('SP1q','TN2q')] = dict(r_min=0.1, U=SP1qxTN2q_coulomb_interaction[0], F=SP1qxTN2q_coulomb_interaction[1])

    if 'SQ4n' in frame.particles.types:
        SQ4n_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=-0.5*0.75, q2=-0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
        SQ4nxTQ5_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=-0.5*0.75, q2=0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
        SQ4nxTN2q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=-0.5*0.75, q2=0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
        # TQ5xSP1q_coulomb_interaction = coulomb_interaction(r_min=0.1, r_max=r_max, q1=0.75, q2=-0.5*0.75, eps_r=eps_r, eps_0=eps_0, table_width=1000)
        table.params[('SQ4n','SQ4n')] = dict(r_min=0.1, U=SQ4n_coulomb_interaction[0],      F=SQ4n_coulomb_interaction[1])
        table.params[('TQ5','SQ4n')] = dict(r_min=0.1,  U=SQ4nxTQ5_coulomb_interaction[0],  F=SQ4nxTQ5_coulomb_interaction[1])
        table.params[('SQ4n','TN2q')] = dict(r_min=0.1, U=SQ4nxTN2q_coulomb_interaction[0], F=SQ4nxTN2q_coulomb_interaction[1])
    

    # Modify the LJ interactions between charged and neutral beads



    #WRITE THIS AFTER WE HAVE AN ANALYSIS THAT GETS US CLOSE TO EXPERIMENTAL VALUES



    mttk = hoomd.md.methods.thermostats.MTTK(kT = kT, tau = 100*0.02)
    cv = hoomd.md.methods.ConstantVolume(filter = hoomd.filter.All(),thermostat = mttk)
    integratorNVT = hoomd.md.Integrator(
        dt=0.02,
        methods=[cv],
        forces=[lj, bond_harmonic, angle_forces,dihedrals],
    )
    sim.operations.integrator = integratorNVT

    sim.run(4e4)
    print("NVT equilibration done, now starting NPT")

    # NPT equilibration
    integratorNVT.forces=[]
    del(integratorNVT)
    npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),S=P,
                                            tauS=1000*0.02,couple='xyz',thermostat=mttk)    
    integratorNPT = hoomd.md.Integrator(dt=0.02,methods=[npt],
                                        forces=[lj,bond_harmonic,angle_forces,dihedrals])
    sim.operations.integrator = integratorNPT

    sim.run(4e5)



if __name__ == "__main__":
    import csv
    main()
