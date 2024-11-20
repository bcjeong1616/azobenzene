import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
import argparse

#------------ Function definitions ---------------
def trim_data(data0, data1):
    # hardcoded that the trimming is based on the first column of data, assumed
    # to be the timesteps
    j = 0
    if len(data0) > len(data1):
        newData = np.zeros((len(data1),len(data0[0])))
        for i in range(len(data0[:,0])):
            if (data0[i][0] == data1[j][0]):
                newData[j,:] = data0[i,:]
                j = j+1
                if j >= len(data1[:,0]):
                    break
        return newData, data1
    else:
        newData = np.zeros((len(data0),len(data1[0])))
        for i in range(len(data1[:,0])):
            if (data0[j][0] == data1[i][0]):
                newData[j,:] = data1[i,:]
                j = j+1
                if j >= len(data0[:,0]):
                    break
        return data0, newData

def calc_true_strain(lx,ly,lz,direction):
    match direction:
        case 'x':
            initial = lx[0]
            trueStrain = np.log(np.divide(lx,initial))
        case 'y':
            initial = ly[0]
            trueStrain = np.log(np.divide(ly,initial))
        case 'z':
            initial = lz[0]
            trueStrain = np.log(np.divide(lz,initial))
    return trueStrain

def set_subplot_axes_lim(ax, matchx=True, matchy=True):
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    for index, subax in np.ndenumerate(ax):
        if (xmin == None) or (subax.get_xlim()[0] < xmin):
            xmin = subax.get_xlim()[0]
        if (xmax == None) or (subax.get_xlim()[1] > xmax):
            xmax = subax.get_xlim()[1]
        if (ymin == None) or (subax.get_ylim()[0] < ymin):
            ymin = subax.get_ylim()[0]
        if (ymax == None) or (subax.get_ylim()[1] > ymax):
            ymax = subax.get_ylim()[1]
    for index, subax in np.ndenumerate(ax):
        if matchx:
            subax.set_xlim([xmin,xmax])
        if matchy:
            subax.set_ylim([ymin,ymax])

# ----------- Graph style formatting -------------
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
scale = 1
dt = 0.005

#----------------- Read input data -------------------

parser = argparse.ArgumentParser(description='Creates plots for log data')
non_opt = parser.add_argument_group('mandatory arguments')
non_opt.add_argument('-i', '--input-list', default=[], nargs='+', dest='input_arr',
                help='Input txt files to plot the data from')

non_opt.add_argument('-y', '--y_headers', default=[], nargs='+', dest='y_headers',
                help='The headers of the data columns to plot')

non_opt.add_argument('-pt','--plot_title', default=[], nargs='+',dest ='plot_title', type=str,
                required = True, help = 'Title for the plot with all data stacked')

opt = parser.add_argument_group('optional arguments')

opt.add_argument('-s','--scale', metavar='<float>',dest ='scale', type=float,
                required = False, help = 'Value to scale all plot dimensions by (default 1.2)')

opt.add_argument('-l', '--data_labels', default=[], nargs='+',
                        dest='data_labels',
                        help='Labels for the data lines IN RESPECTIVE ORDER')

opt.add_argument('-x','--x_header', metavar='<str>',dest ='x_header', type=str,
                required = False, help = 'Header name for the x-axis data')


# -------------------------Parse input options----------------------
args = parser.parse_args()
input_arr = args.input_arr
nInput = len(input_arr)
nLines = len(args.y_headers)
if len(args.data_labels) != 0:
    data_labels = args.data_labels
    if (nLines != len(data_labels)):
        print("__ERROR:_INCONSISTENT_NUMBER_OF_TITLES_OR_INPUT_FILES__")
        exit(1)
else:
    data_labels = args.y_headers

if args.scale != None:
    scale = args.scale

plot_title = ""
for i in args.plot_title:
    plot_title = plot_title + i + ' '

# -------------------------Plot all on the same axis----------------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

# For each input file, create a master data file
for i in range(nInput):
    if i == 0:
        data = np.genfromtxt(str(input_arr[0]),dtype=float, skip_header=1)
    else:
        print('This version of the script does not accomodate multiple input files yet!')
        exit(2)
        data = np.vstack((data,np.genfromtxt(str(input_arr[i]))))
    # params = input_arr[i].split('_')

# Get the column headers from the first line of the txt file
with open(str(input_arr[i])) as f:
    line = f.readline()
    line = line.replace('\n','')
headers = line.split()

#Some formatting to make the headers look nice when used for plotting,
# and also removing any preamble that hoomd 4 adds to its log files
headers[0] = headers[0].replace('#','')
for i,header in enumerate(headers):
    # headers[i] = headers[i].replace('_',' ')
    headers[i] = headers[i].replace('md.compute.ThermodynamicQuantities.','')
    headers[i] = headers[i].replace('Simulation.','')
print("headers:", headers)

# Find the index of the x-axis header
if args.x_header != None:
    x_index = headers.index(args.x_header)
else:
    x_index = 0

# ------------------ Create and plot the data --------------
for i, header in enumerate(headers):
    if header in args.y_headers:
        ax.plot(data[:,x_index],data[:,i],linewidth=2,
                    label=rf'{headers[i]}')

#------------------- Calculate Equilibrium Volume ----------
avg_volume = np.mean(data[-50:-1,headers.index("volume")])
print(avg_volume)
input_gsd = input_arr[0].replace('.txt','.gsd').replace('.log','.gsd')
import gsd.hoomd
traj = gsd.hoomd.open(input_gsd)
n_particles = traj[0].particles.N
number_density = n_particles/avg_volume
print(number_density)
number_density = np.round(number_density,4)
# plt.text(x=data[0,-1],y=data[headers.index("volume"),-1],s=f"number density: {number_density}")
plt.text(x=0,y=avg_volume,s=f"%.2f"%(avg_volume))

ax.plot((data[-500,0],data[-1,0]),(avg_volume,avg_volume),c='Red',linestyle='dashed')

ax.set_ylabel('volume')
# ------------------ Plot Formatting -----------------------
ax.set_xlabel(rf'{headers[x_index]}')
# ax.set_ylabel(rf'Activation Yield')
ax.set_title(rf'{plot_title}',wrap=True,fontsize=14)
# ax.legend(loc='best',fontsize=20,columnspacing=0.1,frameon=False,ncol=1,
#                     labelspacing=0.1,handletextpad=0.5,handlelength=0.6,
#                     draggable=True)
ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
                            bottom=True,top=True,left=True,right=True)
# ax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
# ax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
# ax.set_xlim([0,20*s*dt])
ax.set_ylim([1200,1220])
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                                    # Scientific format for y axis labels

# ----------------- Plot display and saving ---------------
if not (os.path.exists('./plots')):
     os.makedirs('./plots')

plt.tight_layout()
# plot_file_name = plot_title.replace(' ','_')
# plt.savefig('./plots/%s.png'%(plot_file_name))
# plt.savefig('./plots/%s.pdf'%(plot_file_name))
plt.show()
plt.tight_layout