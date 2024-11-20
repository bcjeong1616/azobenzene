#!/opt/miniconda3/bin/python
"""Initialize the project's data space."""
import numpy as np
import signac
import itertools


def grid(gridspec):
    """Yields the Cartesian product of a `dict` of iterables.

    The input ``gridspec`` is a dictionary whose keys correspond to
    parameter names. Each key is associated with an iterable of the
    values that parameter could take on. The result is a sequence of
    dictionaries where each dictionary has one of the unique combinations
    of the parameter values.
    """
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))

# "constant variables"
bAA = 0.9609
bAB = 1.0382
bBB = 1.13
bAS = 1.0   # temporary
bSS = 1.0   # temporary
bBS = 1.0   # temporary

sAA = 1.0
sBB = 1.2
sSS = 0.88
sAB = 0.5*(sAA+sBB)
sAS = 0.5*(sAA+sSS)
sBS = 0.5*(sBB+sSS)

def main():

    project = signac.init_project()

    # first define all parameters of interest in a dictionary (statepoint grid)

    # this script can be changed and simply executed again and again to extend the parameter space
    # new parameters can also be added to the workspace after creating it, executing this script
    # again doesn't delete existing simulations, it only extends the parameter space. The grid
    # does NOT have to be regular, one can iterate/loop over any logic that makes sense for the project.

    # TODO: Add ALL parameters that might be needed and/or varied in simulations
    # this step requires careful planning first!
    statepoint_grid = {
        "replica_index": [0,1,2],#,1,2,3,4], # replica index (if want to run same simulation multiple times for statistics)
        # initialization statepoints
        "volume_fraction": [0.55],
        "n_chains": [1],
        # equilibration statepoints
        "kT": [0.00831446262*298.15],
        "P": [1/16.3882464],
        #system statepoints
        "chain_N": [9,18,27,40],
        "chain_frac_Azo": [0.0],
        "azo_architecture": ["sideChain"], #doesn't matter
        "azo_isomer": ["trans"], #doesn't matter
        "isomerization_scheme": ['all'],
        "n_spacers": [0],
        "mol_frac_IL": [0.95],
        "xlink_frac": [0],
        }

    # loop over all statepoints
    for sp in grid(statepoint_grid):
        # avoid creating any statepoints where the system has main chain Azo and n_spacers > 0
        if(sp.get("azo_architecture") == 'mainChain') and (sp.get("n_spacers") != 0):
            continue

        # open the job and initialize - this will create the "workspace" folders
        job = project.open_job(sp).init()

        # job.sp variables can be calculated as well
        # volA = (sAA/2.)**3*4*np.pi/3.0
        # volB = (sBB/2.)**3*4*np.pi/3.0
        # job.sp['volume_fraction_B'] = (1-job.sp['volume_fraction_A'])
        # job.sp['N_A'] = np.round((volB*job.sp['N_B']/job.sp['volume_fraction_B']-
        #                             volB*job.sp['N_B'])/volA/2.)
        # job.sp['N'] = 2*job.sp['N_A'] + job.sp['N_B']
        # job.sp['N_chains'] = np.round(np.prod(job.sp['box'])*job.sp['volume_fraction']/
        #                                 (2*volA*job.sp['N_A'] + volB*job.sp['N_B']))
        # # job.sp['N_chains'] = np.round(np.prod(job.sp['box'])*job.sp['number_density']/job.sp['N'])


        # define run steps in the job document so that run_steps
        # can be changed without modifying the statepoint
        job.doc['equi_step']=0

        # print something to see what it does - not neccessary
        # print(f"initializing state point with id {job.id}, at density: {job.sp['density']}, fraction {job.sp['fraction_A']} and temperature: {job.sp['temperature']}.")


if __name__ == "__main__":
    main()
