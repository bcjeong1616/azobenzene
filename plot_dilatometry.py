import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os, glob
import string
from matplotlib import transforms as mtransforms
import argparse

#------------ Function definitions ---------------
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
    
#--------------------Parameters of note ----------------------------------------
# direction = 'x'
#-------------------------------------------------------------------------------

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
n_subplot_row = 2
n_subplot_col = 1
dt = 0.005

parser = argparse.ArgumentParser(description='Creates plots for log data')
non_opt = parser.add_argument_group('mandatory arguments')
non_opt.add_argument('-i', '--input-list', default=[], nargs='+',
                        dest='input_arr', help='creates plots from log files')


opt = parser.add_argument_group('optional arguments')
opt.add_argument('-rows','--rows', metavar='<int>',dest ='n_subplot_row', type=int,
                required = False, help = 'Number of subplot rows to create')
opt.add_argument('-cols','--cols', metavar='<int>',dest ='n_subplot_col', type=int,
                required = False, help = 'Number of subplot columns to create')
opt.add_argument('-s','--scale', metavar='<float>',dest ='scale', type=float,
                required = False, help = 'Value to scale all plot dimensions by (default 1.2)')

opt.add_argument('-t', '--input-titles', default=[], nargs='+',
                        dest='input_titles', required = False,
                        help='Titles to be used to label the input files IN RESPECTIVE ORDER')

opt.add_argument('-pt','--plot_title', default=[], nargs='+',dest ='plot_title', type=str,
                required = False, help = 'Title for the plot with all data stacked')



args = parser.parse_args()

input_arr = args.input_arr
nInput = len(input_arr)
if (args.input_titles):
    input_titles = args.input_titles
    if (nInput != len(input_titles)):
        print("__ERROR:_INCONSISTENT_NUMBER_OF_TITLES_OR_INPUT_FILES__")
        exit(1)
else:
    input_titles = args.input_arr
if (args.plot_title):
    plot_title = ""
    for i in args.plot_title:
        plot_title = plot_title + i +' '
    plot_file_name = plot_title.replace(' ','_')
else:
    plot_title = "Dilatometry on " + os.path.basename(str(input_arr[0]).replace('.log',''))
    plot_file_name = plot_title.replace(' ','_').replace("'","").replace('[','').replace(']','')



if args.n_subplot_col != None:
    n_subplot_col = args.n_subplot_col
if args.n_subplot_row != None:
    n_subplot_row = args.n_subplot_row
if args.scale != None:
    scale = args.scale


if (nInput != n_subplot_col*n_subplot_row):
    print("Note: subplot dimensions inconsistent with number of inputs")
    n_subplot_col = nInput
    n_subplot_row = 1

'''
# s = np.zeros(nInput)
# tau = np.zeros(nInput)
# P = np.zeros(nInput)
# tauP = np.zeros(nInput)
# deltaLam = np.zeros(nInput)
'''

'''
with open(input_arr[0], 'r') as fp:
    nLines = len(fp.readlines())

timestep = np.ndarray(shape=(nInput,nLines-1))
temperature = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
pressure = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
kinetic_energy = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
potential_energy = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
volume = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
time = np.ndarray(shape=(nInput,nLines-1))#[[]]*nInput
'''

timestep = []
temperature = []
pressure = []
kinetic_energy = []
potential_energy = []
volume = []
time = []

# pressure_xx = [None]*nInput
# pressure_yy = [None]*nInput
# pressure_zz = [None]*nInput
# lx = [None]*nInput
# ly = [None]*nInput
# lz = [None]*nInput
# true_strains = [None]*nInput

for i,input_log in enumerate(input_arr):
    
    # data = np.genfromtxt(input_log,delimiter = ' ')
    # print(np.shape(data))
    # print(data)
    print(input_log)
    f = open(input_log, "r")
    headers = f.readline()

    #Append another numpy array to the list
    with open(input_arr[0], 'r') as fp:
        nLines = len(fp.readlines())
    timestep.append(np.zeros(shape=nLines,dtype=float))
    temperature.append(np.zeros(shape=nLines,dtype=float))
    pressure.append(np.zeros(shape=nLines,dtype=float))
    kinetic_energy.append(np.zeros(shape=nLines,dtype=float))
    potential_energy.append(np.zeros(shape=nLines,dtype=float))
    volume.append(np.zeros(shape=nLines,dtype=float))
    time.append(np.zeros(shape=nLines,dtype=float))

    j = 0
    for line in f:
        data = line.split()
        # print(data)
        # print(type(data[0]))
        for k,datum in enumerate(data):
            data[k] = float(datum)
        timestep[i][j] = data[0]
        time[i][j] = data[0] * dt
        temperature[i][j] = data[1]
        pressure[i][j] = data[2]
        kinetic_energy[i][j] = data[3]
        potential_energy[i][j] = data[4]
        volume[i][j] = data[5]
        if i == 1 and data[5]< 3.55374e+05:
            print('data[5]: ',data[5])
            print('line: ',line)
            print('data: ',data)
            exit()
        j = j + 1
print(volume)

    # timestep[i] = data[:,0]
    # time[i] = timestep[i]*dt
    # temperature[i] = data[:,1]
    # pressure[i] = data[:,2]
    # kinetic_energy[i] = data[:,3]
    # potential_energy[i] = data[:,4]
    # volume[i] = data[:,5]
    # pressure_xx[i] = data[:,5]
    # pressure_yy[i] = data[:,8]
    # pressure_zz[i] = data[:,10]
    # lx[i] = data[:,11]
    # ly[i] = data[:,12]
    # lz[i] = data[:,13]
    # input_titles[i] = os.path.basename(input_log).replace("_"," ").replace(".log","")
    # input_titles[i] = input_log[input_log.find("xlinked_"):input_log.find("xlinked_")+11]


fig, ax = plt.subplots(n_subplot_row,n_subplot_col,
                figsize=(8.6*scale*cm*n_subplot_col,8.6*scale*cm*n_subplot_row),
                sharex=False,sharey=False)

j = 0
for i,subax in np.ndenumerate(ax):
    # ------------------ Create and plot the data --------------
    #ax.plot(true_strains,pressure_xx*-1,linewidth=2, label = r'$\sigma_{xx}$')
    # volume = np.log(np.multiply(lx[j],np.multiply(ly[j],lz[j])))
    subax.scatter(temperature[j][20::2],np.log(volume[j][20::2]),label=rf'{input_titles[j]}',s=1)
    print(volume[j][1:20])
    print(np.shape(volume))

    # ------------------ Plot Formatting -----------------------
    subax.set_xlabel(r'Temperature')
    subax.set_ylabel(r'ln(Volume)')
    subax.set_title(r'%s'%(input_titles[j]),wrap=True,fontsize=14)
    if nInput > 1:
        subax.legend(loc='best',fontsize=14,columnspacing = 0.1,frameon=False,ncol=1,
                                labelspacing=0.1,handletextpad=0.5,handlelength=0.6,
                                draggable=True)
    subax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                    bottom=True, top=True, left=True, right=True)
    # subax.tick_params(axis='x', which='minor', bottom=True)
    # subax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
    # subax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
    # subax.set_xlim([0,20*s*dt])
    # subax.set_ylim([380001,485000])
    # subax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #Scientific format for y axis labels
    j = j+1
set_subplot_axes_lim(ax)


# plot_title_string = str(input_titles[0])
# for t in input_titles:
#     if t != input_titles[0]:
#         plot_title_string = plot_title_string + "vs" + str(t)
plt.tight_layout()
# plt.savefig('./plots/%s_dilatometry.png'%(os.path.basename(input_log).replace('.log','')))
# plt.savefig('./plots/%s_dilatometry.pdf'%(os.path.basename(input_log).replace('.log','')))

# plt.savefig('./plots/%s_dilatometry1.png'%(plot_file_name))
# plt.savefig('./plots/%s_dilatometry1.pdf'%(plot_file_name))
plt.show()

# -------------------------Plot all on the same axis----------------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

for j in range(nInput):
    # ------------------ Create and plot the data --------------
    # volume = np.log(np.multiply(lx[j],np.multiply(ly[j],lz[j])))
    ax.scatter(temperature[j][20:],np.log(volume[j][20:]),label=rf'{input_titles[j]}',s=1)
    print(volume[j][1:20])

    # ------------------ Plot Formatting -----------------------
    ax.set_xlabel(r'Temperature')
    ax.set_ylabel(r'ln(Volume)')
    ax.set_title(rf'{plot_title}',wrap=True,fontsize=14)
    if nInput > 1:
        ax.legend(loc='best',fontsize=14,columnspacing = 0.1,frameon=False,ncol=1,
            labelspacing=0.1,handletextpad=0.5,handlelength=0.6,draggable=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        bottom=True, top=True, left=True, right=True)
    # ax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
    # ax.set_xlim([0,20*s*dt])
    # ax.set_ylim([380001,485000])
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# ----------------- Plot display and saving ---------------
if not (os.path.exists('./plots')):
     os.makedirs('./plots')

plt.grid()
plt.tight_layout()
# plt.savefig(rf'./plots/{plot_file_name}_dilatometry2.png')
# plt.savefig(rf'./plots/{plot_file_name}_dilatometry2.pdf')
plt.show()
