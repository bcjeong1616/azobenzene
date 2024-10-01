import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gsd
import gsd.hoomd
import argparse
import os
import freud

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
    point_indices = [x for _, x in sorted(zip(query_point_indices, point_indices), 
                                          key=lambda pair: pair[0])]
    query_point_indices = np.sort(query_point_indices)
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
    A_name,
    B_name,
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

        type_A = snap.particles.typeid == snap.particles.types.index(A_name)
        type_B = snap.particles.typeid == snap.particles.types.index(B_name)

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
scale = 2.0
n_subplot_row = 2
n_subplot_col = 1
dt = 0.005

############################### PARSE ##################################

parser = argparse.ArgumentParser(description='Calculate pair correlation function')
non_opt = parser.add_argument_group('mandatory arguments')
# non_opt.add_argument('-i', '--input_list', default=[], nargs='+', dest='input_arr',
#                      required=True, help='creates plots from gsd files')

# non_opt.add_argument('-f', metavar='<int>', dest='frame', type=int,
#                      required=False, help='frame number to start analysis from')
non_opt.add_argument('-t', '--input_titles', default=[], nargs='+',
                        dest='input_titles',
                        help='Titles to be used to label the input files IN RESPECTIVE ORDER')

'''
non_opt.add_argument('-pt','--plot_title', default=[], nargs='+',dest ='plot_title', type=str,
                required = True, help = 'Title for the plot with all data stacked')
'''

non_opt.add_argument('-o', metavar='<str>', dest='output_prefix', type=str,
                     required=False, help='output file prefix for pair correlation function result')

args = parser.parse_args()
# input_arr = args.input_arr
input_arr = ["./data/box_of_azo/traj.gsd","./data/box_of_azo/eq_iso_traj.gsd"]
nInput = len(input_arr)
if args.input_titles != None:
    input_titles = args.input_titles
else:
    input_titles = np.empty()

'''
plot_title = ''
for i in args.plot_title:
    plot_title = plot_title + i +' '
plot_file_name = plot_title.replace(' ','_')
'''

if (nInput != n_subplot_row*n_subplot_col):
    print("subplot dimensions INVALID")
    n_subplot_row = 1
    n_subplot_col = nInput

# if (nInput != len(input_titles)):
#     print("__ERROR:_INCONSISTENT_NUMBER_OF_TITLES_OR_INPUT_FILES__")
#     exit(1)

# -----------------------------------------------------------------------
#               Figure out input and output
# -----------------------------------------------------------------------

# input_directory = os.path.dirname(args.infile)

# output_figure_path = os.path.join(input_directory, 'rdf_figure.png')
# output_txt_path = os.path.join(input_directory, args.outfile)



# in_file = gsd.fl.GSDFile(name=input_arr, mode='r', application='hoomd', schema='hoomd', schema_version=[1, 0])
# trajectory = gsd.hoomd.HOOMDTrajectory(in_file)

nbins = 300
rmax = 4*0.34
rmin = 0.1*0.34

# rdf = freud.density.RDF(bins=nbins, r_max=rmax, r_min=rmin)
# rdf, normalization = intermolecular_rdf(
#     input_arr, A_name='A', B_name='A', r_max=rmax, exclude_bonded=True, bins=nbins, r_min=rmin, start=-5
# )

fig, ax = plt.subplots(3,2,
                figsize=(8.6*scale*cm*n_subplot_col,8.6*scale*cm*n_subplot_row),
                sharex=False,sharey=False)
j = 0
for i,subax in np.ndenumerate(ax):
    # ------------------ Create and plot the data --------------
    if i[0] == 0:
        A_name='TC5'
        B_name='TC5'
    elif i[0] == 1:
        A_name='TC5'
        B_name='TN3r'
    elif i[0] == 2:
        A_name='TN3r'
        B_name='TN3r'
    rdf, normalization = intermolecular_rdf(
        input_arr[i[1]], A_name=A_name, B_name=B_name, exclude_bonded=True, 
        bins=nbins, start=0, r_max=rmax, r_min=rmin
    )
    subax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2)
    # ------------------ Plot Formatting -----------------------
    if i[0] == 2:
        subax.set_xlabel(rf'r [nm]',fontsize=16)
    subax.set_ylabel(r'$g_{%s}$(r)'%(A_name[1] +':'+ B_name[1]),fontsize=16)
    if i[0] == 0:
        if i[1] == 0:
            conformation = 'trans-AB'
        else:
            conformation = 'cis-AB'
        subax.set_title(rf'{conformation}',wrap=True,fontsize=20)
    subax.legend(loc='best',fontsize=16,columnspacing = 0.1,frameon=False,ncol=1,
                            labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    subax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                    bottom=True, top=True, left=True, right=True)
    subax.tick_params(axis='x', which='minor', bottom=True)
    subax.set_xticks([0.4,0.8,1.2])
    subax.set_yticks([0.0,0.5,1.0])
    subax.set_xlim([0.3,1.3])
    # subax.set_ylim([380001,485000])
    # j = j+1
    subax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
set_subplot_axes_lim(ax)

if not (os.path.exists('./plots')):
    os.makedirs('./plots')

plt.tight_layout()
plt.savefig('./plots/Peter_%s_RDFs.png'%('azo_box'))
plt.savefig('./plots/Peter_%s_RDFs.pdf'%('azo_box'))
plt.show()

'''
# -------------------------Plot all on the same axis----------------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

for i in range(nInput):
    # ------------------ Create and plot the data --------------
    rdf, normalization = intermolecular_rdf(
        input_arr[i], A_name='TN3r', B_name='TN3r', r_max=rmax, exclude_bonded=True,
        bins=nbins, r_min=rmin, start=0
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
'''

#Plot on the same plot as Peter's
fig, ax = plt.subplots(2,2,
                figsize=(8.6*scale*cm*n_subplot_col,8.6*scale*cm*n_subplot_row),
                sharex=False,sharey=False)
j = 0
for i,subax in np.ndenumerate(ax):
    if (i[1] == 0):
        #Don't plot in first column 
        continue
    # ------------------ Create and plot the data --------------
    if i[0] == 0:
        A_name='TN3r'
        B_name='TN3r'
    elif i[0] == 1:
        A_name='TC5'
        B_name='TC5'
    for k in range(len(input_arr)):
        rdf, normalization = intermolecular_rdf(
            input_arr[k], A_name=A_name, B_name=B_name, exclude_bonded=True, 
            bins=nbins, start=0, r_max=rmax, r_min=rmin
        )
        if k == 0:
            conformation = 'trans-AB'
        else:
            conformation = 'cis-AB' 
        subax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=conformation)

# Plot isomerization effect of peters
peter_trans_NN_x = [0.33164556962025316, 0.3569620253164557, 0.3822784810126582, 0.40759493670886077, 0.43291139240506327, 0.4582278481012658, 0.4962025316455696, 0.5531645569620254, 0.6101265822784809, 0.6734177215189874, 0.7113924050632912, 0.7556962025316456, 0.8063291139240507, 0.8506329113924049, 0.90126582278481, 0.9645569620253165, 1.0341772151898734, 1.1037974683544303, 1.160759493670886, 1.2240506329113925, 1.2873417721518987]
peter_trans_NN_y = [0.15725806451612903, 0.32661290322580644, 0.4838709677419355, 0.6532258064516129, 0.7862903225806451, 0.907258064516129, 1.0161290322580645, 1.0524193548387097, 1.0161290322580645, 1.1129032258064515, 1.2338709677419355, 1.379032258064516, 1.4032258064516128, 1.3064516129032258, 1.221774193548387, 1.185483870967742, 1.1975806451612903, 1.2096774193548387, 1.221774193548387, 1.1975806451612903, 1.1975806451612903]

peter_cis_NN_x = [0.3189873417721519, 0.34430379746835443, 0.36962025316455693, 0.3949367088607595, 0.42658227848101266, 0.45189873417721516, 0.4772151898734177, 0.4962025316455696, 0.5151898734177215, 0.5594936708860759, 0.5974683544303797, 0.6417721518987342, 0.6734177215189874, 0.6987341772151898, 0.7240506329113924, 0.7493670886075949, 0.8189873417721518, 0.7936708860759494, 0.8506329113924049, 0.90126582278481, 0.9518987341772152, 0.9962025316455696, 1.0531645569620254, 1.110126582278481, 1.160759493670886, 1.2240506329113925, 1.2873417721518987]
peter_cis_NN_y = [0.0967741935483871, 0.22983870967741934, 0.3508064516129032, 0.4596774193548387, 0.5806451612903225, 0.689516129032258, 0.8346774193548386, 0.9798387096774194, 1.0887096774193548, 1.161290322580645, 1.125, 1.1008064516129032, 1.161290322580645, 1.2459677419354838, 1.3548387096774193, 1.4516129032258065, 1.342741935483871, 1.4153225806451613, 1.2459677419354838, 1.1975806451612903, 1.2096774193548387, 1.1733870967741935, 1.1491935483870968, 1.161290322580645, 1.185483870967742, 1.1975806451612903, 1.1975806451612903]

peter_trans_PP_x = [0.3314465408805031, 0.35660377358490564, 0.3754716981132076, 0.4006289308176101, 0.41949685534591197, 0.43836477987421385, 0.4509433962264151, 0.46352201257861636, 0.4761006289308176, 0.48867924528301887, 0.5075471698113208, 0.5452830188679245, 0.5955974842767295, 0.620754716981132, 0.6396226415094339, 0.6522012578616352, 0.6647798742138364, 0.6836477987421383, 0.7025157232704402, 0.740251572327044, 0.7842767295597484, 0.8220125786163521, 0.8597484276729559, 0.8974842767295597, 0.9477987421383649, 0.9981132075471699, 1.0547169811320756, 1.1176100628930818, 1.1742138364779875, 1.2245283018867925, 1.2748427672955975]
peter_trans_PP_y = [0.028, 0.168, 0.28, 0.406, 0.532, 0.672, 0.812, 0.9380000000000001, 1.092, 1.246, 1.358, 1.442, 1.414, 1.316, 1.204, 1.106, 0.98, 0.882, 0.77, 0.686, 0.728, 0.812, 0.91, 0.98, 1.05, 1.064, 1.05, 1.036, 1.022, 0.994, 0.966]

peter_cis_PP_x = [0.35000000000000003, 0.3875, 0.41250000000000003, 0.4375, 0.45625000000000004, 0.46875, 0.48125, 0.48750000000000004, 0.50625, 0.51875, 0.54375, 0.5875, 0.61875, 0.64375, 0.66875, 0.7, 0.7375, 0.7875, 0.8374999999999999, 0.88125, 0.93125, 1.0187499999999998, 0.975, 1.075, 1.13125, 1.18125, 1.2374999999999998, 1.29375]
peter_cis_PP_y = [0.02777777777777779, 0.125, 0.25, 0.375, 0.5277777777777778, 0.6805555555555556, 0.8472222222222222, 1, 1.1527777777777777, 1.2638888888888888, 1.375, 1.375, 1.2916666666666665, 1.1805555555555554, 1.0416666666666665, 0.9166666666666666, 0.8333333333333333, 0.861111111111111, 0.9305555555555555, 0.986111111111111, 1.0277777777777777, 1.0416666666666665, 1.0416666666666665, 1.0138888888888888, 1, 1, 1, 1]

ax[0][0].plot(peter_trans_NN_x, peter_trans_NN_y, linewidth=2, label="trans-AB")
ax[0][0].plot(peter_cis_NN_x, peter_cis_NN_y, linewidth=2, label="cis-AB")
ax[1][0].plot(peter_trans_PP_x, peter_trans_PP_y, linewidth=2, label="trans-AB")
ax[1][0].plot(peter_cis_PP_x, peter_cis_PP_y, linewidth=2, label="cis-AB")

# ------------------ Plot Formatting -----------------------
for i,subax in np.ndenumerate(ax):
    subax.legend(loc='best',fontsize=16,columnspacing = 0.1,frameon=False,ncol=1,
                            labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    if i[0] == 0:
        #if first row
        title = "Peter et al" if (i[1] == 0) else "Martini Model"
        subax.set_title(rf'{title}',wrap=True,fontsize=20)

    if i[0] == 1:
        #if last row
        subax.set_xlabel(rf'r [nm]',fontsize=16)

    if i[1] == 0:
        #if first column
        pair = "NN" if (i[0] == 0) else "CC"
        subax.set_ylabel(r'$g_{%s}$(r)'%(pair),fontsize=16)
    subax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                    bottom=True, top=True, left=True, right=True)
    subax.tick_params(axis='x', which='minor', bottom=True)
    subax.set_xticks([0.4,0.8,1.2])
    subax.set_yticks([0.0,0.5,1.0,1.5])
    subax.set_xlim([0.3,1.3])
    # subax.set_ylim([380001,485000])
    # j = j+1
    subax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))

set_subplot_axes_lim(ax)

plt.tight_layout()
plt.show()

#Plot on the same plot as Peter's
fig, ax = plt.subplots(1,2,
                figsize=(8.6*scale*cm*n_subplot_col,8.6*scale*cm*n_subplot_row),
                sharex=False,sharey=False)
j = 0
for i,subax in np.ndenumerate(ax):
    if (i[1] == 0):
        #Don't plot in first column 
        continue
    # ------------------ Create and plot the data --------------
    A_name='TN3r'
    B_name='TN3r'
    for k in range(len(input_arr)):
        rdf, normalization = intermolecular_rdf(
            input_arr[k], A_name=A_name, B_name=B_name, exclude_bonded=True, 
            bins=nbins, start=0, r_max=rmax, r_min=rmin
        )
        if k == 0:
            conformation = 'trans-AB'
        else:
            conformation = 'cis-AB'
        subax.plot(rdf.bin_centers, rdf.rdf * normalization, linewidth=2, label=conformation)

# Plot isomerization effect of peters
peter_trans_NN_x = [0.33164556962025316, 0.3569620253164557, 0.3822784810126582, 0.40759493670886077, 0.43291139240506327, 0.4582278481012658, 0.4962025316455696, 0.5531645569620254, 0.6101265822784809, 0.6734177215189874, 0.7113924050632912, 0.7556962025316456, 0.8063291139240507, 0.8506329113924049, 0.90126582278481, 0.9645569620253165, 1.0341772151898734, 1.1037974683544303, 1.160759493670886, 1.2240506329113925, 1.2873417721518987]
peter_trans_NN_y = [0.15725806451612903, 0.32661290322580644, 0.4838709677419355, 0.6532258064516129, 0.7862903225806451, 0.907258064516129, 1.0161290322580645, 1.0524193548387097, 1.0161290322580645, 1.1129032258064515, 1.2338709677419355, 1.379032258064516, 1.4032258064516128, 1.3064516129032258, 1.221774193548387, 1.185483870967742, 1.1975806451612903, 1.2096774193548387, 1.221774193548387, 1.1975806451612903, 1.1975806451612903]

peter_cis_NN_x = [0.3189873417721519, 0.34430379746835443, 0.36962025316455693, 0.3949367088607595, 0.42658227848101266, 0.45189873417721516, 0.4772151898734177, 0.4962025316455696, 0.5151898734177215, 0.5594936708860759, 0.5974683544303797, 0.6417721518987342, 0.6734177215189874, 0.6987341772151898, 0.7240506329113924, 0.7493670886075949, 0.8189873417721518, 0.7936708860759494, 0.8506329113924049, 0.90126582278481, 0.9518987341772152, 0.9962025316455696, 1.0531645569620254, 1.110126582278481, 1.160759493670886, 1.2240506329113925, 1.2873417721518987]
peter_cis_NN_y = [0.0967741935483871, 0.22983870967741934, 0.3508064516129032, 0.4596774193548387, 0.5806451612903225, 0.689516129032258, 0.8346774193548386, 0.9798387096774194, 1.0887096774193548, 1.161290322580645, 1.125, 1.1008064516129032, 1.161290322580645, 1.2459677419354838, 1.3548387096774193, 1.4516129032258065, 1.342741935483871, 1.4153225806451613, 1.2459677419354838, 1.1975806451612903, 1.2096774193548387, 1.1733870967741935, 1.1491935483870968, 1.161290322580645, 1.185483870967742, 1.1975806451612903, 1.1975806451612903]

ax[0][0].plot(peter_trans_NN_x, peter_trans_NN_y, linewidth=2, label="trans-AB")
ax[0][0].plot(peter_cis_NN_x, peter_cis_NN_y, linewidth=2, label="cis-AB")

# ------------------ Plot Formatting -----------------------
for i,subax in np.ndenumerate(ax):
    subax.legend(loc='best',fontsize=16,columnspacing = 0.1,frameon=False,ncol=1,
                            labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
    if i[0] == 0:
        #if first row
        title = "Peter et al" if (i[1] == 0) else "Martini Model"
        subax.set_title(rf'{title}',wrap=True,fontsize=20)

    if i[0] == 1:
        #if last row
        subax.set_xlabel(rf'r [nm]',fontsize=16)

    if i[1] == 0:
        #if first column
        pair = "NN" if (i[0] == 0) else "CC"
        subax.set_ylabel(r'$g_{%s}$(r)'%(pair),fontsize=16)
    subax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                    bottom=True, top=True, left=True, right=True)
    subax.tick_params(axis='x', which='minor', bottom=True)
    subax.set_xticks([0.4,0.8,1.2])
    subax.set_yticks([0.0,0.5,1.0,1.5])
    subax.set_xlim([0.3,1.3])
    # subax.set_ylim([380001,485000])
    # j = j+1
    subax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))

set_subplot_axes_lim(ax)

plt.tight_layout()
plt.show()