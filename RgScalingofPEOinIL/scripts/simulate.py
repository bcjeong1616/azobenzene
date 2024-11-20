import numpy as np
import gsd,gsd.hoomd
import sys, os 
sys.path.insert(0,"/home/bj21/updateAzplugins/hoomd-4.8.2-install/")
import hoomd
import hoomd.azplugins
sys.path.append('/home/bj21/programs/')
from martini3 import molecules
from martini3 import init_cell
from martini3 import force_fields
import random

from timeit import default_timer as timer
from scripts.system import System

def wrap_pbc(x,box):
        delta = np.where(x > 0.5 * box, x - box, x)
        delta = np.where(delta < - 0.5 * box, box + delta, delta)
        return delta

def gen_grid_3D(low_bound, up_bound, num_spaces=100):
    grid = []
    x_arr = np.linspace(low_bound, up_bound, num_spaces)
    for x in x_arr:
        for y in x_arr:
            for z in x_arr:
                grid.append((x, y, z))
    return grid

import datetime
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

class Simulator():
    def __init__(self,job):
        self.mol_frac_IL = job.sp["mol_frac_IL"]
        self.job = job

        
        # self.kT = job.sp['temperature']
        # self.rho = job.sp['density']

        # file names 
        self.init_gsd_file = job.fn('init.gsd')
        self.equi_gsd_file = job.fn('equi.gsd')

    def initialize_system(self):
        # contents setup
        contents = molecules.Contents()

        kT = 300*0.0083144626181532 # convert T to kT, 25C
        # IL_mol_frac_arr = [0.99]#0.005 #[0.4,1.0]#0.0 [0.1,0.2,
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

        num_ILs = int(round(self.mol_frac_IL*total_mol))
        num_chains = 1
        grid = gen_grid_3D(low_bound, up_bound, num_spaces=int(x_box_size*1.5))

        #create the initial gsd to run the simulation from
        #first, the chain

        #random position
        new_pos = random.choice(grid)
        #random orientation
        alpha = np.random.rand()*2*np.pi
        beta  = np.random.rand()*2*np.pi
        gamma = np.random.rand()*2*np.pi
        
        PEOx9 = molecules.make_polymer(
            name="PEOx9",
            contents=contents,
            x_shift=new_pos[0],
            y_shift=new_pos[1],
            z_shift=new_pos[2],
            alpha=alpha,
            beta=beta,
            gamma=gamma
            )
        #Give the chain a random walk configuration
        print(PEOx9.positions)
        PEOx9.randomize_linear()
        print(PEOx9.positions)
        #account for PBC:
        molecule_position = np.array(PEOx9.positions)
        # print(np.shape(PEOx9.position))
        for j,position in enumerate(PEOx9.positions):
            molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
        PEOx9.positions = molecule_position

        contents.add_molecule(PEOx9)
        grid.remove(new_pos)

        for i in range(num_ILs):
            #random position
            new_pos = random.choice(grid)
            #random orientation
            alpha = np.random.rand()*2*np.pi
            beta  = np.random.rand()*2*np.pi
            gamma = np.random.rand()*2*np.pi

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
            
            #Add EMIM
            #random position
            new_pos = random.choice(grid)
            #random orientation
            alpha = np.random.rand()*2*np.pi
            beta  = np.random.rand()*2*np.pi
            gamma = np.random.rand()*2*np.pi
            
            EMIM = molecules.make_molecule(
                name="EMIM",
                contents=contents,
                x_shift=new_pos[0],
                y_shift=new_pos[1],
                z_shift=new_pos[2],
                alpha=alpha,
                beta=beta,
                gamma=gamma
                )
            #account for PBC:
            molecule_position = np.array(EMIM.positions)
            # print(np.shape(lithium.position))
            for j,position in enumerate(EMIM.positions):
                molecule_position[j] = wrap_pbc(position,box=np.array([x_box_size,x_box_size,z_box_size]))
            EMIM.positions = molecule_position

            contents.add_molecule(EMIM)
            grid.remove(new_pos)

        #initialize the gsd
        path = f"workspace/{self.job.id}/"
        lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = init_cell.init_cell(
            contents, path, box_size=[x_box_size, x_box_size, z_box_size], pair_on=False #False
        )
        # init cell also saves the gsd
        print("Initialization for N=",self.job.sp["chain_N"], " completed, starting equilibration")

    def equilibrate(self,init_path,kT,P):
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

        gsd_file = init_path + 'equi.gsd'
        log_file = init_path + 'equi.log'

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
                            forces=[dpd, bond_harmonic, angle_forces],)
        fire.forces.extend(dihedrals)

        sim.operations.integrator = fire
        sim.run(1e5)

        print("FIRE 1 done")

        for A in np.linspace(10,400,5):
            print("A: ",A)
            dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=A, gamma=4.5)
            fire.forces = [dpd, bond_harmonic, angle_forces]
            fire.forces.extend(dihedrals)
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
        bjerrum_length = 1 / (4 * np.pi * eps_0 * eps_r)
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
            forces=[lj, bond_harmonic, angle_forces],
        )
        integratorNVT.forces.extend(dihedrals)
        sim.operations.integrator = integratorNVT

        sim.run(4e4)
        print("NVT equilibration done, now starting NPT")

        # NPT equilibration
        integratorNVT.forces=[]
        del(integratorNVT)
        npt = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),S=P,
                                                tauS=1000*0.02,couple='xyz',thermostat=mttk)    
        integratorNPT = hoomd.md.Integrator(dt=0.02,methods=[npt],
                                        forces=[lj,bond_harmonic,angle_forces])
        integratorNPT.forces.extend(dihedrals)
        sim.operations.integrator = integratorNPT

        sim.run(4e5)

    def polymerize(self):
        if "custom_action" in self.job.sp["polymerization_method"]:
            import scripts.customAction.polymerize as polymerize
        
        S = System()

        # try:
        #     cpu = hoomd.device.GPU(notice_level=5)
        # except:
        #     cpu = hoomd.device.CPU(notice_level=5)
        if self.job.sp["polymerization_method"] == "custom_action_GPU" or self.job.sp["polymerization_method"] == "custom_action_GPU_bulk":
            device = hoomd.device.GPU(notice_level=3)
        else:
            device = hoomd.device.CPU(notice_level=3)
        print(f"job_id: {self.job.id}",self.job.sp["polymerization_method"]," is being run on ",device)

        sim = hoomd.Simulation(device=device, seed=1)

        if os.path.isfile(self.polymerize_gsd_file) and os.path.getsize(self.polymerize_gsd_file)>1e4:
            sim.create_state_from_gsd(filename=self.polymerize_gsd_file)
        else: 
            sim.create_state_from_gsd(filename=self.equi_gsd_file)
        
        if len(sim.state.angle_types)==0:
            FJ_system=True
        else:
            FJ_system=False

        integrator = hoomd.md.Integrator(dt=0.005)
        sim.operations.integrator = integrator 

        cell = hoomd.md.nlist.Cell(buffer=0.4)

        lj = hoomd.md.pair.LJ(nlist=cell)
        lj.params[(S.particles_types, S.particles_types)] = dict(epsilon=1.0,sigma=1.0)
        lj.r_cut[S.particles_types, S.particles_types] = 2.5

        lj.params[(S.particles_types, 'Dummy')] = dict(epsilon=0.0,sigma=0.0)
        lj.r_cut[S.particles_types, 'Dummy'] = 0
        lj.mode = 'shift'

        fene = hoomd.md.bond.FENEWCA()
        fene.params[S.bond_types] = dict(k=30,r0=1.5,epsilon=1.0, sigma=1.0, delta=0.0)
        fene.params['Dummy'] = dict(k=0,r0=1.5,epsilon=0.0, sigma=1.0, delta=0.0)

        if FJ_system==False:
            cosinesq = hoomd.md.angle.CosineSquared()
            cosinesq.params[S.angle_types] = dict(k=self.angle_constant,    t0=np.pi*110/180)# https://www.sciencedirect.com/science/article/pii/S0032386110003642?ref=cra_js_challenge&fr=RR-1
            cosinesq.params['Dummy'] = dict(k=0.0001, t0=np.pi)  # k>0 to make warning go away (should not do anything)

            integrator.forces = [lj,fene,cosinesq]
        else:
            integrator.forces = [lj,fene]

        types_to_integrate =  hoomd.filter.Type(S.particles_types[:-1])

       

        if os.path.isfile(self.polymerize_gsd_file) and os.path.getsize(self.polymerize_gsd_file)>1e4:
            # already has radicals in polymerize.gsd 
            print("Continue reaction...")
        else:
            print("Starting reaction...")
            # flip some sulfurs to be radical, i.e. "initialization by photoinitiator"
            snapshot = sim.state.get_snapshot()
            S.flip_radicals_on(snapshot,self.radical_number_percent)
            sim.state.set_snapshot(snapshot)
        
        npt = hoomd.md.methods.ConstantPressure(
            filter=types_to_integrate,
            tauS=1000*sim.operations.integrator.dt,
            gamma=2/(1000*sim.operations.integrator.dt),
            S=0.0,
            couple="xyz",
            rescale_all=True,
            thermostat=hoomd.md.methods.thermostats.MTTK(kT=self.kT,tau=sim.operations.integrator.dt*100))
        
        sim.state.thermalize_particle_momenta(filter=types_to_integrate, kT=self.kT)
        #zero_momentum = hoomd.md.update.ZeroMomentum( hoomd.trigger.On(1000))
        #sim.operations.updaters.append(zero_momentum)
        
        sim.operations.integrator.methods.append(npt)

        # Define and add the GSD operation.
        gsd_writer = hoomd.write.GSD(filename=self.polymerize_gsd_file,
                                    trigger=hoomd.trigger.Periodic(100*100),#self.polymerize_period),
                                    dynamic=['property','momentum','topology','attribute'],
                                    mode='ab')
        sim.operations.writers.append(gsd_writer)
        print("writing gsd file to:",self.polymerize_gsd_file)
        thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=types_to_integrate)
        sim.operations.computes.append(thermodynamic_properties)
       
        status = Status(sim)
        logger = hoomd.logging.Logger(categories=['scalar','string'])
       
        logger.add(sim, quantities=['timestep','tps'])
        logger.add(thermodynamic_properties, quantities=['kinetic_temperature','pressure','kinetic_energy','potential_energy','volume'])

        # table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(self.polymerize_period,100),logger=logger)
        # table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(self.polymerize_period,100),logger=logger, output=open(self.output_txt,'a'))
        table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(100,100),logger=logger)
        table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(100,100),logger=logger, output=open(self.output_txt,'a'))
        sim.operations.writers.append(table_stdout)
        sim.operations.writers.append(table_file)

        if "custom_action" in self.job.sp["polymerization_method"]:
            if self.job.sp["polymerization_method"] == "custom_action_GPU" or self.job.sp["polymerization_method"] == "custom_action_CPU" :
                #define and add the actions and operations to keep track of extent of reaction
                extent_of_reaction_action = polymerize.calc_extent_of_reaction(
                                            simulator=self) 
                extent_of_reaction_operation = hoomd.update.CustomUpdater(
                    action=extent_of_reaction_action, trigger=self.polymerize_period
                )
                sim.operations += extent_of_reaction_operation

                #define and add the polymerization actions and operations
                bond_formation_action = polymerize.bond_formation(
                                            probability=1.0,r_cut=self.r_cut_reaction,
                                            FJ_system=FJ_system) 
                    #thiol reaction probability defaults to 1.0 in system's propagate reactions
                bond_formation_operation = hoomd.update.CustomUpdater(
                    action=bond_formation_action, trigger=self.polymerize_period
                )
                sim.operations += bond_formation_operation

                # polymerize.chain_growth(self.)
                chain_growth_action = polymerize.chain_growth(
                                            probability=self.chain_side_reaction_probability,
                                            r_cut=self.r_cut_reaction,FJ_system=FJ_system)
                chain_growth_operation = hoomd.update.CustomUpdater(
                    action=chain_growth_action, trigger=self.polymerize_period
                )
                sim.operations += chain_growth_operation

                # polymerize.chain_transfer(self.)
                chain_transfer_action = polymerize.chain_transfer(
                                                probability=self.chain_transfer_probability,
                                                r_cut=self.r_cut_reaction,FJ_system=FJ_system) 
                chain_transfer_operation = hoomd.update.CustomUpdater(
                    action=chain_transfer_action, trigger=self.polymerize_period
                )
                sim.operations += chain_transfer_operation

                '''
                termination_action = polymerize.termination_reactions(
                                        probability=1.0,r_cut=self.r_cut_reaction,FJ_system=FJ_system)
                    #thiol reaction probability defaults to 1.0 in system's propagate reactions
                termination_operation = hoomd.update.CustomUpdater(
                    action=termination_action, trigger=self.polymerize_period
                )
                sim.operations += termination_operation
                '''
            elif self.job.sp["polymerization_method"] == "custom_action_GPU_bulk":
                bulk_polymerization_action = polymerize.bulk_polymerize(
                                        r_cut=self.r_cut_reaction,FJ_system=FJ_system,simulator=self)
                bulk_polymerization_operation = hoomd.update.CustomUpdater(
                    action=bulk_polymerization_action, trigger=self.polymerize_period
                )
                sim.operations += bulk_polymerization_operation


            polymerization_times = []
            integration_times = []
            integration_tps = []
            # tps = []

            simulation_start = timer()
            
            for i in range(10000):
                period_start = timer()

                polymerization_start = timer()
                sim.run(1)
                polymerization_end = timer()
                polymerization_times.append(polymerization_end - polymerization_start)

                integration_start = timer()
                sim.run(self.polymerize_period-1)
                integration_end = timer()
                integration_times.append(integration_end - integration_start)

                integration_tps.append((self.polymerize_period-1)/integration_times[-1])

                gsd_writer.flush()

                period_end = timer()
                # tps.append(period_end - period_start)

                # self.job.doc["TPS"] = np.average(tps)
                self.job.doc["avg_polymerization_time"] = np.average(polymerization_times)
                self.job.doc["avg_integration_time"] = np.average(integration_times)
                self.job.doc["integration_tps"] = np.average(integration_tps)

                if self.job.doc["reacted_monomers"] > 0.925:
                    exit()
        else:
            if "local_snapshot" not in self.job.sp["polymerization_method"]:
                print("ERROR: Polymerization method not recognized")
                exit(2)
            polymerization_times = []
            integration_times = []
            integration_tps = []

            for i in range(10000):
                propagate_start = timer()
                with sim.state.cpu_local_snapshot as snapshot:
                    S.propagate_reaction(snapshot,
                                        r_cut=self.r_cut_reaction,
                                        chain_transfer_probability=self.chain_transfer_probability,
                                        chain_side_reaction_probability=self.chain_side_reaction_probability)
                    
                    dummy_id = len(sim.state.bond_types)-1
                    bonds = snapshot.bonds.group[snapshot.bonds.typeid!=dummy_id] # remove dummy type bond 
                    nids, counts = np.unique(np.concatenate(bonds).flatten(),return_counts=True)
                    bonded = nids[counts>=2]
                    #reacted_monomers = bonded[snapshot.bonds.typeid[bonded]==1]

                    ids = np.arange(len(snapshot.particles.tag))
                    idx = snapshot.particles.rtag[ids]
                    particle_ids = snapshot.particles.typeid[idx]

                    unreacted_enes = len(ids[particle_ids==1])/2
                    self.job.doc['reacted_monomers'] = 1 - unreacted_enes/(self.N_monomers*2)

                propagate_end = timer()

                sim.run(self.polymerize_period)

                gsd_writer.flush()

                period_end = timer()

                polymerization_times.append(propagate_end - propagate_start)
                integration_times.append(period_end - propagate_end)
                integration_tps.append(self.polymerize_period/(period_end - propagate_end))

                self.job.doc["avg_polymerization_time"] = np.average(polymerization_times)
                self.job.doc["avg_integration_time"] = np.average(integration_times)
                self.job.doc["integration_tps"] = np.average(integration_tps)

                if self.job.doc["reacted_monomers"] > 0.925:
                    exit()

    # def run(self):

    #     S = System()
        
    #     try:
    #         cpu = hoomd.device.GPU()
    #     except:
    #         cpu = hoomd.device.CPU()
        
    #     sim = hoomd.Simulation(device=cpu, seed=1)
    #     sim.create_state_from_gsd(filename=self.polymerize_gsd_file)

    #     if len(sim.state.angle_types)==0:
    #         FJ_system=True
    #     else:
    #         FJ_system=False

    #     integrator = hoomd.md.Integrator(dt=0.005)
    #     sim.operations.integrator = integrator

    #     cell = hoomd.md.nlist.Cell(buffer=0.4)

    #     lj = hoomd.md.pair.LJ(nlist=cell)
    #     lj.params[(S.particles_types, S.particles_types)] = dict(epsilon=1.0,sigma=1.0)
    #     lj.r_cut[S.particles_types, S.particles_types] = 2.5

    #     lj.params[(S.particles_types, 'Dummy')] = dict(epsilon=0.0,sigma=0.0)
    #     lj.r_cut[S.particles_types, 'Dummy'] = 0
    #     lj.mode = 'shift'

    #     fene = hoomd.md.bond.FENEWCA()
    #     fene.params[S.bond_types] = dict(k=30,r0=1.5,epsilon=1.0, sigma=1.0, delta=0.0)
    #     fene.params['Dummy'] = dict(k=0,r0=1.5,epsilon=0.0, sigma=1.0, delta=0.0)

    #     if FJ_system==False:
    #         harmonic_a = hoomd.md.angle.Harmonic()
    #         harmonic_a.params[S.angle_types] = dict(k=self.angle_constant,    t0=np.pi)
    #         harmonic_a.params['Dummy'] = dict(k=1e-6, t0=np.pi)  # k>0 to make warning go away (should not do anything)

    #         integrator.forces = [lj,fene,harmonic_a]
    #     else: 
    #         integrator.forces = [lj,fene]

    #     types_to_integrate =  hoomd.filter.Type(S.particles_types[:-1])

    #     npt = hoomd.md.methods.ConstantPressure(
    #         filter=types_to_integrate,
    #         tauS=1000*sim.operations.integrator.dt,
    #         gamma=2/(1000*sim.operations.integrator.dt),
    #         S=0.0,
    #         couple="xyz",
    #         rescale_all=True,
    #         thermostat=hoomd.md.methods.thermostats.MTTK(kT=self.kT,tau=sim.operations.integrator.dt*100))
        
    #     sim.state.thermalize_particle_momenta(filter=types_to_integrate, kT=self.kT)
    #     #zero_momentum = hoomd.md.update.ZeroMomentum( hoomd.trigger.On(1000))
    #     #sim.operations.updaters.append(zero_momentum)
        
    #     sim.operations.integrator.methods.append(npt)

    #     # Define and add the GSD operation.
    #     gsd_writer = hoomd.write.GSD(filename=self.run_gsd_file,
    #                                 trigger=hoomd.trigger.Periodic(1000),
    #                                 dynamic=['property','momentum','topology','attribute'],
    #                                 mode='wb')
    #     sim.operations.writers.append(gsd_writer)

    #     thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=types_to_integrate)
    #     sim.operations.computes.append(thermodynamic_properties)
       
    #     logger = hoomd.logging.Logger(categories=['scalar'])
       
    #     logger.add(sim, quantities=['timestep'])
    #     logger.add(thermodynamic_properties, quantities=['kinetic_temperature','pressure','kinetic_energy','potential_energy','volume'])

    #     table_stdout = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger)
    #     table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000),logger=logger, output=open(self.output_run_txt,'w'))
    #     sim.operations.writers.append(table_stdout)
    #     sim.operations.writers.append(table_file)

    #     sim.run(10000000)



