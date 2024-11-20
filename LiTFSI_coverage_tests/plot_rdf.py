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
            #Change the position of PEO dimers to be in between the two
            if A_name == 'SN3r':
                A_pos = average_positions(A_pos, 2)
            if B_name == 'SN3r':
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
n_subplot_row = 2
n_subplot_col = 1
dt = 0.005

############################### PARSE ##################################

parser = argparse.ArgumentParser(description='Calculate pair correlation function')
non_opt = parser.add_argument_group('mandatory arguments')
non_opt.add_argument('-i', '--input_list', default=[], nargs='+', dest='input_arr',
                help='creates plots from gsd files')

# non_opt.add_argument('-f', metavar='<int>', dest='frame', type=int,
#                      required=False, help='frame number to start analysis from')
non_opt.add_argument('-t', '--input_titles', default=[], nargs='+',
                        dest='input_titles',
                        help='Titles to be used to label the input files IN RESPECTIVE ORDER')

non_opt.add_argument('-pt','--plot_title', default=[], nargs='+',dest ='plot_title', type=str,
                required = True, help = 'Title for the plot with all data stacked')

non_opt.add_argument('-o', metavar='<str>', dest='output_prefix', type=str,
                     required=False, help='output file prefix for pair correlation function result')

args = parser.parse_args()
input_arr = args.input_arr
nInput = len(input_arr)
input_titles = args.input_titles

plot_title = ''
for i in args.plot_title:
    plot_title = plot_title + i +' '
plot_file_name = plot_title.replace(' ','_')

if (nInput != n_subplot_row*n_subplot_col):
    print("subplot dimensions INVALID")
    n_subplot_row = 1
    n_subplot_col = nInput

if (nInput != len(input_titles)):
    print("__ERROR:_INCONSISTENT_NUMBER_OF_TITLES_OR_INPUT_FILES__")
    exit(1)

# -----------------------------------------------------------------------
#               Figure out input and output
# -----------------------------------------------------------------------

# input_directory = os.path.dirname(args.infile)

# output_figure_path = os.path.join(input_directory, 'rdf_figure.png')
# output_txt_path = os.path.join(input_directory, args.outfile)



# in_file = gsd.fl.GSDFile(name=input_arr, mode='r', application='hoomd', schema='hoomd', schema_version=[1, 0])
# trajectory = gsd.hoomd.HOOMDTrajectory(in_file)

nbins = 300
rmax = 2.5
rmin = 0.1

# rdf = freud.density.RDF(bins=nbins, r_max=rmax, r_min=rmin)
# rdf, normalization = intermolecular_rdf(
#     input_arr, A_name='A', B_name='A', r_max=rmax, exclude_bonded=True, bins=nbins, r_min=rmin, start=-5
# )

fig, ax = plt.subplots(n_subplot_row,n_subplot_col,
                figsize=(8.6*scale*cm*n_subplot_col,8.6*scale*cm*n_subplot_row),
                sharex=False,sharey=False)
j = 0
for i,subax in np.ndenumerate(ax):
    # ------------------ Create and plot the data --------------
    rdf, normalization = intermolecular_rdf(
        input_arr[j], A_name='SQ4n', B_name='TQ5', r_max=rmax, exclude_bonded=True, 
        bins=nbins, r_min=rmin, start=-20
    )
    subax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2)
    # ------------------ Plot Formatting -----------------------
    subax.set_xlabel(rf'r [$\sigma$]')
    subax.set_ylabel(rf'g(r)')
    subax.set_title(r'RDF from %s'%(str(input_titles[j])),wrap=True,fontsize=14)
    subax.legend(loc='best',fontsize=16,columnspacing = 0.1,frameon=False,ncol=1,
                            labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    subax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                    bottom=True, top=True, left=True, right=True)
    subax.tick_params(axis='x', which='minor', bottom=True)
    # subax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
    # subax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
    # subax.set_xlim([0,20*s*dt])
    # subax.set_ylim([380001,485000])
    j = j+1
set_subplot_axes_lim(ax)

if not (os.path.exists('./plots')):
    os.makedirs('./plots')

plt.tight_layout()
plt.savefig('./plots/%s_RDF1.png'%(plot_file_name))
plt.savefig('./plots/%s_RDF1.pdf'%(plot_file_name))

# -------------------------Plot all on the same axis----------------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

# xlink_arr = ["1.2","1.35"]
# xlink_arr = [1.2,1.35,1.5,2]
# label_arr = ["Quartic","FENE"]
for i in range(nInput):
    # ------------------ Create and plot the data --------------
    rdf, normalization = intermolecular_rdf(
        input_arr[i], A_name='TQ5', B_name='SN3r', r_max=rmax, exclude_bonded=False,
        bins=nbins, r_min=rmin, start=-20
    )
    ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=rf'{input_titles[i]}')
    # ------------------ Plot Formatting -----------------------
    ax.set_xlabel(rf'r [$\sigma$]')
    ax.set_ylabel(rf'g(r)')
    ax.set_title(rf'{plot_title}',wrap=True,fontsize=14)
    ax.legend(loc='best',fontsize=20,columnspacing = 0.1,frameon=False,ncol=1,
                        labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
                                bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    # ax[i].set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax[i].set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax[i].set_xlim([0,20*s*dt])
    # ax[i].set_ylim([380001,485000])
    # ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                        #Scientific format for y axis labels

plt.tight_layout()
plt.savefig('./plots/%s_RDF2.png'%(plot_file_name))
plt.savefig('./plots/%s_RDF2.pdf'%(plot_file_name))

np.savetxt(rf'{plot_file_name}.txt', np.c_[rdf.bin_centers, rdf.rdf], header="bincenter, RDF")
plt.show()


# -------------------------Plot partial RDFs all on the same axis----------------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

# xlink_arr = ["1.2","1.35"]
# xlink_arr = [1.2,1.35,1.5,2]
# label_arr = ["Quartic","FENE"]
for i in range(nInput):
    # ------------------ Create and plot the data --------------
    rdf, normalization = intermolecular_rdf(
        input_arr[i], A_name='TQ5', B_name='TQ5', r_max=rmax, exclude_bonded=True,
        bins=nbins, r_min=rmin, start=-20
    )
    ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=rf'TQ5-TQ5')

    rdf, normalization = intermolecular_rdf(
        input_arr[i], A_name='TQ5', B_name='SQ4n', r_max=rmax, exclude_bonded=True,
        bins=nbins, r_min=rmin, start=-20
    )
    ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=rf'TQ5-SQ4n')

    rdf, normalization = intermolecular_rdf(
        input_arr[i], A_name='TQ5', B_name='SN3r', r_max=rmax, exclude_bonded=True,
        bins=nbins, r_min=rmin, start=-20
    )
    ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=rf'TQ5-SN3r')

    # rdf, normalization = intermolecular_rdf(
    #     input_arr[i], A_name='SP1q', B_name='SP1q', r_max=rmax, exclude_bonded=True,
    #     bins=nbins, r_min=rmin, start=-20
    # )
    # ax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=rf'SP1q-SP1q')
    # ------------------ Plot Formatting -----------------------
    ax.set_xlabel(rf'r [$\sigma$]')
    ax.set_ylabel(rf'g(r)')
    ax.set_title(rf'Partial RDFs of Li',wrap=True,fontsize=14)
    ax.legend(loc='best',fontsize=20,columnspacing = 0.1,frameon=False,ncol=1,
                        labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
                                bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='x', which='minor', bottom=True)
    # ax[i].set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax[i].set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax[i].set_xlim([0,20*s*dt])
    # ax[i].set_ylim([380001,485000])
    # ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                        #Scientific format for y axis labels

plt.tight_layout()
plt.savefig('./plots/%s_RDF2.png'%(plot_file_name))
plt.savefig('./plots/%s_RDF2.pdf'%(plot_file_name))

np.savetxt(rf'{plot_file_name}.txt', np.c_[rdf.bin_centers, rdf.rdf], header="bincenter, RDF")
plt.show()