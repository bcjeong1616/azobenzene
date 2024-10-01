import numpy as np
import sys
sys.path.append('/home/bj21/programs/')
from martini3 import molecules
from martini3 import init_cell
from martini3 import force_fields
import hoomd
import os
import math
import datetime

def gen_grid(low_bound, up_bound, num_spaces=100):
    grid = []
    x_arr = np.linspace(low_bound, up_bound, num_spaces)
    for x in x_arr:
        for y in x_arr:
            grid.append((x, y))
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

    path = "data/" + "box_of_azo" +"/"
    if not os.path.exists(path):
        os.makedirs(path)
    # contents setup
    contents = molecules.Contents()

    density = 0.925 #g/cm**3
    total_azos = 300
    azo_mass = 182.226 * 1.66054*10**-24 # convert AMU to grams
    volume = total_azos*azo_mass/density 
    volume = volume * 10**21            # convert cm**3 to nm**3
    x_box_size = volume**(1/3)
    z_box_size = volume**(1/3)
    # print(x_box_size)

    azos_placed = 0
    positions = np.empty((0, 3))
    np.random.seed()
    low_bound = -x_box_size / 2 + .25
    up_bound = x_box_size / 2 - .25
    low_bound_z = -z_box_size / 2 + .5
    up_bound_z = z_box_size/2 -.5

    grid_polym = gen_grid(low_bound, up_bound, num_spaces=int(x_box_size * 50))
    z_grid = np.linspace(low_bound_z, up_bound_z, int(z_box_size * 400))
    # print(len(grid_polym))

    num_molecule_beads = 0
    while azos_placed < total_azos:
        # 1. generate a coordinate and rotation
        # 2. pick if we want to place polym or lipid
        # 3. attempt to place and compare distance
        # 4. if placed, add position to list
        polym_percent = azos_placed / total_azos
        # lipid_percent = lipid_placed/total_lipids
        is_inverted = bool(np.random.randint(0, 2))
        z_shift = z_grid[int(np.random.randint(0, len(z_grid)))]
        
        # print("len(grid_polym):",len(grid_polym))
        rand_pos = int(np.random.randint(0, len(grid_polym)))
        x_shift, y_shift = grid_polym[rand_pos]
        # molecule = molecules.make_PCLbPEO(contents,5,10,x_shift = x_shift,y_shift = y_shift, z_shift = z_shift, is_inverted=is_inverted)

        #random orientations
        alpha = np.random.rand()*2*math.pi
        beta  = np.random.rand()*2*math.pi
        gamma = np.random.rand()*2*math.pi
        molecule = molecules.make_azobenzene(
            contents,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
            alpha=alpha,
            beta=beta,
            gamma=gamma
            )
        
        del grid_polym[rand_pos]
        molecule_position = np.array(molecule.position)
        # print(np.shape(molecule.position))
        for j,position in enumerate(molecule.position):
            molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
        molecule.position = molecule_position
        azos_placed += 1
        num_molecule_beads+=7
        contents.add_molecule(molecule)
        positions = np.append(positions, molecule_position, axis=0)
        '''
        # print("molecule_position = ", molecule_position)
        # print("First boolean: ", (
        #     np.all(
        #         np.linalg.norm(molecule_position[:, np.newaxis, :] - positions, axis=2)
        #         > 0.43
        #     )))
        # print("second boolean: ", (np.all(
        #         (molecule_position[:,2])**2 <low_bound_z**2) and np.all(
        #         (molecule_position[:,1])**2 <low_bound**2) and np.all((molecule_position[:,0])**2 <low_bound**2)))
        if (len(positions) == 0 or (
            np.all(
                np.linalg.norm(molecule_position[:, np.newaxis, :] - positions, axis=2)
                > 0.43
            ))) and  (np.all(
                (molecule_position[:,2])**2 <low_bound_z**2) and np.all(
                (molecule_position[:,1])**2 <low_bound**2) and np.all((molecule_position[:,0])**2 <low_bound**2)):
            num_molecule_beads+=1
            contents.add_molecule(molecule)
            print(positions)
            positions = np.append(positions, molecule_position, axis=0)
            print(positions)
            azos_placed = azos_placed + 1
            # print("if statement entered")
            # print(contents.contents[0].position)
        '''

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = init_cell.init_cell(
        contents, path, box_size=[x_box_size, x_box_size, z_box_size], pair_on=False
    )
    # init cell also saves the gsd

    path = "data/" + "box_of_azo" +"/"
    if not os.path.exists(path):
        raise Exception("no data present")

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = force_fields.forces_from_gsd(
        path, "init.gsd"
    )

    try:
        sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=16)
        print("Running on the GPU")
    except:
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=16)
        print("Running on the CPU")

    name = "init.gsd"

    sim.create_state_from_gsd(filename=path + name)

    status = Status(sim)
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(sim, quantities=["timestep", "tps"])
    logger[("Status", "etr")] = (status, "etr", "string")
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(period=50000), logger=logger
    )
    sim.operations.writers.append(table)

    import gsd
    with gsd.hoomd.open(name=path + name, mode="r") as f:
        frame = f[0]
        nl = hoomd.md.nlist.Cell(buffer=0.4)
        dpd = hoomd.md.pair.DPD(nlist=nl, kT=1.5, default_r_cut=1.0)
        dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=10, gamma=4.5)


    # apply FIRE minimization to remove overlaps
    mttk = hoomd.md.methods.thermostats.MTTK(kT = 1.5, tau = 1)
    cv = hoomd.md.methods.ConstantVolume(filter = hoomd.filter.All(),thermostat = mttk)
    # langevin = hoomd.md.methods.Langevin(filter = hoomd.filter.All(),kT=2.47)
    fire = hoomd.md.minimize.FIRE(dt=0.0005,
                        force_tol=1e-2,
                        angmom_tol=1e-2,
                        energy_tol=1e-7,
                        methods=[cv],
                        forces=[dpd, bond_harmonic, angle_forces],)
    
    sim.operations.integrator = fire
    sim.run(1e5)

    print("FIRE 1 done")

    for A in np.linspace(10,400,5):
        print("A: ",A)
        dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=A, gamma=4.5)
        fire.forces = [dpd, bond_harmonic, angle_forces]
        sim.operations.integrator = fire
        sim.run(1e5)

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = force_fields.forces_from_gsd(
        path, "init.gsd"
    )
    mttk = hoomd.md.methods.thermostats.MTTK(kT = 3.3258, tau = 1)
    cv = hoomd.md.methods.ConstantVolume(filter = hoomd.filter.All(),thermostat = mttk)
    integratorNVT = hoomd.md.Integrator(
        dt=0.02,
        methods=[cv],
        forces=[lj, bond_harmonic, angle_forces],
    )
    sim.operations.integrator = integratorNVT

    #-----------------------------------------------------------------------
    #                               Output
    #-----------------------------------------------------------------------

    gsd_file = path + 'traj.gsd'
    log_file = path + 'traj.log'

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

    sim.run(4e6)



if __name__ == "__main__":
    import csv
    main()
