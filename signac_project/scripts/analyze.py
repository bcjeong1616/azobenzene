import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gsd
import gsd.hoomd
import argparse
import os
import freud

def average_positions(pos, group_size):
    if len(pos)%group_size != 0:
        print("Error: position array must be divisible by the number of particles in the group")
        exit(2)
    n_groups = int(len(pos)/group_size)
    avg_positions = np.zeros(shape=(n_groups,3))
    for i in range(n_groups):
        avg_positions[i] = np.mean(pos[i:i+group_size], axis=0)
    return avg_positions

def set_subplot_axes_lim(ax, matchx=True, matchy=True):
    j = 0
    for i,subax in np.ndenumerate(ax):
        if j == 0:
            xmin = subax.get_xlim()[0]
            xmax = subax.get_xlim()[1]
            ymin = subax.get_ylim()[0]
            ymax = subax.get_ylim()[1]

        if subax.get_xlim()[0] < xmin:
            xmin = subax.get_xlim()[0]
        if subax.get_xlim()[1] > xmax:
            xmax = subax.get_xlim()[1]
        if subax.get_ylim()[0] < ymin:
            ymin = subax.get_ylim()[0]
        if subax.get_ylim()[1] > ymax:
            ymax = subax.get_ylim()[1]
        j = j+1
    for i,subax in np.ndenumerate(ax):
        if matchx:
            subax.set_xlim([xmin,xmax])
        if matchy:
            subax.set_ylim([ymin,ymax])

def snap_molecule_indices(snap):
    """Find molecule index for each particle.

    Given a snapshot from a trajectory, compute clusters of bonded molecules
    and return an array of the molecule index of each particle.

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot
        Trajectory snapshot.

    Returns
    -------
    numpy array (N_particles,)

    """
    system = freud.AABBQuery.from_system(snap)
    num_query_points = num_points = snap.particles.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    # print(np.shape(query_point_indices))
    # print(np.shape(point_indices))
    # This may fix the error of the indices not being sorted, but uncertain
    # sorted_indices = np.argsort(query_point_indices)
    # query_point_indices = query_point_indices[sorted_indices]
    # point_indices = point_indices[sorted_indices]
    vectors = system.box.wrap(
        system.points[query_point_indices] - system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, vectors
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx

def intermolecular_rdf(
    gsdfile,
    A_name=None,
    B_name=None,
    start=0,
    stop=None,
    r_max=None,
    r_min=0,
    bins=100,
    exclude_bonded=True
):
    """Compute intermolecular RDF from a GSD file.

    This function calculates the radial distribution function given a GSD file
    and the names of the particle types. By default it will calculate the RDF
    for the entire trajectory.

    It is assumed that the bonding, number of particles, and simulation box do
    not change during the simulation.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name : str
        Name(s) of particles between which to calculate the RDF (found in
        gsd.hoomd.Snapshot.particles.types)
    start : int
        Starting frame index for accumulating the RDF. Negative numbers index
        from the end. (Default value = 0)
    stop : int
        Final frame index for accumulating the RDF. If None, the last frame
        will be used. (Default value = None)
    r_max : float
        Maximum radius of RDF. If None, half of the maximum box size is used.
        (Default value = None)
    r_min : float
        Minimum radius of RDF. (Default value = 0)
    bins : int
        Number of bins to use when calculating the RDF. (Default value = 100)
    exclude_bonded : bool
        Whether to remove particles in same molecule from the neighbor list.
        (Default value = True)

    Returns
    -------
    freud.density.RDF
    """
    with gsd.hoomd.open(gsdfile) as trajectory:
        snap = trajectory[0]

        if r_max is None:
            # Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.5, 0, dtype=np.float32
            )

        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)

        if A_name == None and B_name == None:
            type_A = [True]*snap.particles.N
            type_B = [True]*snap.particles.N
        else:
            try:
                type_A = snap.particles.typeid == snap.particles.types.index(A_name)
            except:
                type_A = snap.particles.typeid == snap.particles.types.index('SQ4n')
            # print(type_A)
            # print(len(type_A))
            # print(snap.particles.N)
            # exit()
            try:
                type_B = snap.particles.typeid == snap.particles.types.index(B_name)
            except:
                type_B = snap.particles.typeid == snap.particles.types.index('SQ4n')

        if exclude_bonded:
            molecules = snap_molecule_indices(snap)
            molecules_A = molecules[type_A]
            molecules_B = molecules[type_B]

        for snap in trajectory[start:stop]:
            A_pos = snap.particles.position[type_A]
            if A_name == B_name:
                B_pos = A_pos
                exclude_ii = True
            else:
                B_pos = snap.particles.position[type_B]
                exclude_ii = False

            # print(A_pos)
            # print(np.shape(A_pos))
            #Change the position of TFSI partial anions to be in between the two
            if A_name == 'SP1q':
                A_pos = average_positions(A_pos, 2)
            if B_name == 'SP1q':
                B_pos = average_positions(B_pos, 2)
            if A_name == 'SQ4n':
                A_pos = average_positions(A_pos, 2)
            if B_name == 'SQ4n':
                B_pos = average_positions(B_pos, 2)
            #Change the position of EMIM partial cations to be in between the two
            if A_name == 'TN2q':
                A_pos = average_positions(A_pos, 2)
            if B_name == 'TN2q':
                B_pos = average_positions(B_pos, 2)
            # print(A_pos)
            # print(np.shape(A_pos))


            box = snap.configuration.box
            system = (box, A_pos)
            aq = freud.locality.AABBQuery.from_system(system)
            nlist = aq.query(
                B_pos, {"r_max": r_max, "exclude_ii": exclude_ii}
            ).toNeighborList()

            if exclude_bonded:
                pre_filter = len(nlist)
                indices_A = molecules_A[nlist.point_indices]
                indices_B = molecules_B[nlist.query_point_indices]
                nlist.filter(indices_A != indices_B)
                post_filter = len(nlist)

            rdf.compute(aq, neighbors=nlist, reset=False)
        normalization = post_filter / pre_filter if exclude_bonded else 1
        return rdf, normalization

# ----------- Graph style formatting -------------
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['text.usetex'] = False
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'sans'
matplotlib.rcParams['mathtext.it'] = 'sans:italic'
matplotlib.rcParams['mathtext.default'] = 'it'
plt.rcParams.update({'axes.linewidth': 1.5})
plt.rcParams.update({'xtick.major.width': 2.5})
plt.rcParams.update({'xtick.direction': 'in'})
plt.rcParams.update({'ytick.direction': 'in'})
plt.rcParams.update({'ytick.major.width': 2.5})
plt.rcParams.update({'errorbar.capsize': 1.5})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
cm = 1/2.54
scale = 1.0
# n_subplot_row = 2
# n_subplot_col = 1
dt = 0.02

class Analyzer():
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
        if not (os.path.exists(self.job.fn('plots'))):
            os.makedirs(self.job.fn('plots'))

    def plot_rdf_cation_anion(self):
        plot_file_name = self.job.fn('plots/rdf_cation_anion.pdf')
        plot_title = 'RDF of Cation-Anion'

        nbins = 300
        rmax = 2.5
        rmin = 0.1

        fig, ax = plt.subplots(1,1,
                        figsize=(8.6*scale*cm,8.6*scale*cm),
                        sharex=False,sharey=False)

        # ------------------ Create and plot the data --------------
        rdf, normalization = intermolecular_rdf(
            gsdfile=self.job.fn('exist.gsd'), A_name='SQ4n', B_name='TN2q', r_max=rmax,
            exclude_bonded=False, bins=nbins, r_min=rmin, start=-20
            # exclude_bonded=True, bins=nbins, r_min=rmin, start=-20
        )
        ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2)
        # ------------------ Plot Formatting -----------------------
        ax.set_xlabel(rf'r [$\sigma$]')
        ax.set_ylabel(rf'g(r)')
        ax.set_title(plot_title,wrap=True,fontsize=14)
        ax.legend(loc='best',fontsize=16,columnspacing = 0.1,frameon=False,ncol=1,
                                labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                        bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', bottom=True)
        # ax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
        # ax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
        # ax.set_xlim([0,20*s*dt])
        # ax.set_ylim([380001,485000])

        plt.tight_layout()
        plt.savefig(plot_file_name)
        np.savetxt(rf'{plot_file_name.replace('pdf','txt')}', np.c_[rdf.bin_centers, rdf.rdf], header="bincenter, RDF", fmt='%f')