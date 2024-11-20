import matplotlib
import matplotlib.pyplot as plt
import argparse
import gsd.hoomd
import numpy as np
import math
import os
from collections import defaultdict
import freud

#-----------------------------------------------------------------------
#                          Function Definitions
#-----------------------------------------------------------------------

def average_positions(pos, group_size):
    if len(pos)%group_size != 0:
        print("Error: position array must be divisible by the number of particles in the group")
        exit(2)
    n_groups = int(len(pos)/group_size)
    avg_positions = np.zeros(shape=(n_groups,3))
    for i in range(n_groups):
        avg_positions[i] = np.mean(pos[i:i+group_size], axis=0)
    return avg_positions

def dihedral_pbc(x0,x1,x2,x3,Box):
    x10 = x1 - x0
    for i in range(len(x10)):
        if np.abs(x10[i]) > 0.5*Box[i]:
            if x10[i] > 0:
                x10[i] = x10[i] - Box[i]
            else:
                x10[i] = x10[i] + Box[i]
    x21 = x2 - x1
    for i in range(len(x21)):
        if np.abs(x21[i]) > 0.5*Box[i]:
            if x21[i] > 0:
                x21[i] = x21[i] - Box[i]
            else:
                x21[i] = x21[i] + Box[i]
    x32 = x3 - x2
    for i in range(len(x32)):
        if np.abs(x32[i]) > 0.5*Box[i]:
            if x32[i] > 0:
                x32[i] = x32[i] - Box[i]
            else:
                x32[i] = x32[i] + Box[i]
    a = np.cross(x32,x21)
    b = np.cross(x21,x10)
    dihedral = math.acos(-np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
    return dihedral

def angle_pbc(x0,x1,x2,Box):
    x01 = x0 - x1
    for i in range(len(x01)):
        if np.abs(x01[i]) > 0.5*Box[i]:
            if x01[i] > 0:
                x01[i] = x01[i] - Box[i]
            else:
                x01[i] = x01[i] + Box[i]

    x21 = x2 - x1
    for i in range(len(x21)):
        if np.abs(x21[i]) > 0.5*Box[i]:
            if x21[i] > 0:
                x21[i] = x21[i] - Box[i]
            else:
                x21[i] = x21[i] + Box[i]

    cosine_angle = np.dot(x01, x21) / (np.linalg.norm(x01) * np.linalg.norm(x21))
    angle = np.arccos(cosine_angle)
    return angle

def dist_pbc(x1,x0,Box):
    delta = np.abs(x1 - x0)
    delta= np.where(delta > 0.5 * Box, Box - delta, delta)
    return np.sqrt(np.sum(delta**2.0))

def connected_components(lists):
    R"""
    merges lists with common elements

    args: bonds from configuration (frame.bonds.group)

    returns: list of connected particles by id

    Useful for finding bonded particles in a configuration, tested for
    linear polymers with consecutive bonds (0-1-2-3-4-5, 6-7-8-9-10,..)
    and non consecutive ids ( 0-5-6-8-10, 1-4-3-2-9,...) but no other
    configuration yet. Works with ints as well as str.

    """
    neighbors = defaultdict(set)
    seen = set()
    for each in lists:
        for item in each:
            neighbors[item].update(each)
    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            see(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield sorted(component(node))

matplotlib.rcParams['font.size'] = 18
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
scale = 1.2

#----------------- Read input data -------------------
parser = argparse.ArgumentParser(description='Creates plots for log data')
non_opt = parser.add_argument_group('mandatory arguments')
# non_opt.add_argument('-i', '--input-list', default=[], nargs='+', dest='input_arr',
#                 help='creates plots from gsd files')

# non_opt.add_argument('-f', '--frame', type=int, dest='frame_index',
#                         help='index of gsd frame to analyze')

# non_opt.add_argument('-t', '--input-titles', default=[], nargs='+',
#                         dest='input_titles',
#                         help='Titles to be used to label the input files IN RESPECTIVE ORDER')

# non_opt.add_argument('-pt','--plot_title', default=[], nargs='+',dest ='plot_title', type=str,
#                 required = True, help = 'Title for the plot with all data stacked')

opt = parser.add_argument_group('optional arguments')
# opt.add_argument('-rows','--rows', metavar='<int>',dest ='n_subplot_row', type=int,
#                 required = False, help = 'Number of subplot rows to create')
# opt.add_argument('-cols','--cols', metavar='<int>',dest ='n_subplot_col', type=int,
#                 required = False, help = 'Number of subplot columns to create')
opt.add_argument('-s','--scale', metavar='<float>',dest ='scale', type=float,
                required = False, help = 'Value to scale all plot dimensions by (default 1.2)')

args = parser.parse_args()
# input_arr = args.input_arr
# nInput = len(input_arr)
# input_titles = args.input_titles
# if (nInput != len(input_titles)):
#     print("__ERROR:_INCONSISTENT_NUMBER_OF_TITLES_OR_INPUT_FILES__")
#     exit(1)

if args.scale != None:
    scale = args.scale

# plot_title = ""
# for i in args.plot_title:
#     plot_title = plot_title + i +' '
# plot_file_name = plot_title.replace(' ','_')

import os
subdirs = [x[0] for x in os.walk('./data/')]
print(subdirs)
input_arr = subdirs[1:]
input_arr = [x + '/traj.gsd' for x in input_arr]
print(input_arr)

salt_mol_frac_arr = []
frac_CIPs_arr = []
for input_gsd in input_arr:
    dihedralsList = np.array([])
    traj = gsd.hoomd.open(input_gsd,mode='r')
    salt_mol_frac = os.path.dirname(input_gsd).split(sep='_')[-1]
    print("salt_mol_frac:",salt_mol_frac)
    if float(salt_mol_frac) == 0.0:
        #no TFSI to test the coverage of
        continue
    salt_mol_frac_arr.append(float(salt_mol_frac))

    frac_CIPs = []
    for frame in traj[-50:-1]:
        box = frame.configuration.box[0:3]
        part_typeids = frame.particles.typeid
        pos = frame.particles.position
        dihedralGroups = frame.dihedrals.group

        #down-select the positions of TFSI anions
        if 'SQ4n' in frame.particles.types:
            tfsi_anion_type = frame.particles.types.index('SQ4n')
        elif 'SP1q' in frame.particles.types:
            tfsi_anion_type = frame.particles.types.index('SP1q')
        else:
            print("Error: unknown TFSI anion bead type")
            exit()
        anion_type_mask = frame.particles.typeid == tfsi_anion_type
        anion_pos = pos[anion_type_mask]
        #make anion positions the average of the two in the tfsi
        anion_pos = average_positions(anion_pos, 2)

        #down-select the positions of Li cations
        li_type_mask = frame.particles.typeid == frame.particles.types.index('TQ5')
        cation_pos = pos[li_type_mask]

        #for each TFSI, query if there is at least one Li within the CIP cutoff
        r_max = 1.5 * 0.365*2**(1/6)
        box = frame.configuration.box
        system = (box, anion_pos)
        aq = freud.locality.AABBQuery.from_system(system)
        nlist = aq.query(cation_pos, {"mode":'nearest',"num_neighbors": 1, "r_max": r_max, "exclude_ii": True}).toNeighborList()

        frac_CIPs.append(len(nlist)/len(anion_pos))
    frac_CIPs_arr.append(np.mean(frac_CIPs))

print(salt_mol_frac_arr)
print(frac_CIPs_arr)
#Plot the list of dihedrals as a histogram
plt.scatter(salt_mol_frac_arr,np.subtract(1,frac_CIPs_arr))
plt.xlabel(rf'Salt mol fraction')
plt.ylabel(rf'Fraction of Free/SSIPs')
plt.tight_layout()
plt.show()