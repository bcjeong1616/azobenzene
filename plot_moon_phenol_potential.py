import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math

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
scale = 2

# ------------------ Create and plot the data --------------
fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

name_arr = ['Moon et al. 9-6','Martini 3 TC5']
eps_arr = [0.476*4.184,1.51] #Moon et al is in kcal, so converted here
sig_arr = [6.55*0.1,0.34] #Moon et al is in angstroms, so converted here

r_arr = np.linspace(0.1,2,500)

'''force_divr_arr = np.array([0]*500,dtype=float)'''
for i,name in enumerate(name_arr):
    eng_arr = np.array([0]*500,dtype=float)
    match name:
            case 'Moon et al. 9-6':
                def potential(eps,sig,r):
                    prefactor = eps*(sig/r)**6
                    diff = 2*(sig/r)**3 - 3
                    return prefactor*diff
                # def force()
            case 'Martini 3 TC5':
                def potential(eps,sig,r):
                    sig_divr_6 = (sig/r)**6
                    prefactor = 4*eps*sig_divr_6
                    diff = sig_divr_6 - 1
                    return prefactor*diff
    for j,r in enumerate(r_arr):
        eng_arr[j] = potential(eps=eps_arr[i],sig=sig_arr[i],r=r)



    # #Find the point at which the force is 0
    # eps = 0.1
    # equi_r = r_arr[np.where(np.logical_and(force_divr_arr < eps,force_divr_arr > -eps))]
    # print(equi_r)
    # exit()

    '''
    # print(r_arr)
    # print(bond_eng_arr)
    '''
    ax.plot(r_arr,eng_arr,linewidth=2,label=rf'{name}')
    # ax[1].plot(r_arr,force_divr_arr,linewidth=2,label=rf'Force/r')

# ------------------ Plot Formatting -----------------------
ax.set_xlabel(rf'r [nm]')
ax.set_ylabel(rf'Bond Energy [kJ/mol]')
ax.set_title(rf'Self-Interaction Potentials of Azobenzene Benzene Group',wrap=True,fontsize=14)
ax.legend(loc='best',fontsize=20,columnspacing=0.1,frameon=False,ncol=1,
                    labelspacing=0.1,handletextpad=0.5,handlelength=0.6,draggable=True)
ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
                            bottom=True,top=True,left=True,right=True)
# ax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
# ax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
ax.set_xlim([0,1.5])
ax.set_ylim([-10,100])
ax.grid()
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                    #Scientific format for y axis labels


# ----------------- Plot display and saving ---------------
# if not (os.path.exists('./plots')):
#     os.makedirs('./plots')

plt.tight_layout()
# plot_file_name = plot_title.replace(' ','_')
# plt.savefig('./plots/%s.png'%(plot_file_name))
# plt.savefig('./plots/%s.pdf'%(plot_file_name))
plt.show()
'''
    #plot the colormap energy -----------------------------
    fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
                sharex=False,sharey=False)

    X, Y = np.meshgrid(r_arr, theta_arr)
    Z = np.log(((V_max-c_half)/(b_sq*b_sq))*((X-a/2)**2-b_sq)**2 + c_half/b*(X-a/2) + c_half + 2*1/2*k*(X/2-r0_arr[i]/2)**2 + 1/2*angle_k*(Y-math.pi)**2)
    contour = ax.contourf(X,Y,Z,cmap="Blues",levels = 20)
    cbar = fig.colorbar(contour)
    cbar.ax.set_ylabel(rf'$ln(Total Energy)$')
    contourlines = ax.contour(contour, levels=contour.levels[::2], colors='r')
    # ax[0].scatter(r_arr,theta_arr,c=bond_angle_eng_arr.flatten(),cmap="Blues",linewidth=2,label=rf'Bond Energy')
    # ax.colorbar()
    ax.set_xlim([0.5,2])
    ax.set_ylim([math.pi/2,3/2*math.pi])
    ax.set_xlabel(rf'r [$\sigma$]')
    ax.set_ylabel(rf'Bond Angle [rad]')
    plt.tight_layout()
    plt.show()

# #----------------------------------------------------------------------------------
# #                         Test new parameters for glassy MP
# #----------------------------------------------------------------------------------

# # ------------------ Create and plot the data --------------
# fig, ax = plt.subplots(1,2,figsize=(8.6*scale*cm,8.6*scale*cm),
#             sharex=False,sharey=False)

# V_max_arr = [3.0,2.7]
# b = 0.2
# c_arr = [0.5,-0.7]
# a = 2*0.96+2*b
# name_arr = ['old_AA_DW','new_AA_DW']
# # b_arr = [0.2,0.2,0.2]
# # c_arr = [0.5,0.5,0.5]

# k_arr = [15,45]
# r0 = 0.96

# angle_k = 50

# r_arr = np.linspace(0.1,2,500)
# # theta_arr = np.linspace(math.pi/2,3/2*math.pi,500)

# bond_eng_arr = np.array([0]*500,dtype=float)
# force_divr_arr = np.array([0]*500,dtype=float)
# bond_angle_eng_arr = np.ndarray(shape=(500,500),dtype=float)
# for i,name in enumerate(name_arr):
#     for j,r in enumerate(r_arr):
#         #For calculating the doublewell contribution
#         c_half = (0.5)*c_arr[i]
#         r_min_half_a = r-(0.5)*a
#         b_sq = b*b
#         d = r_min_half_a*r_min_half_a - b_sq;
#         bond_eng_arr[j] = ((V_max_arr[i]-c_half)/(b_sq*b_sq))*d*d + c_half/b*r_min_half_a + c_half
#         force_divr_arr[j] = - (4*(V_max_arr[i]-c_half)/(b_sq*b_sq)*d*r_min_half_a+c_half/b)/r

#         #For calculating the harmonic bond potential contribution
#         bond_eng_arr[j] = bond_eng_arr[j] + 2*1/2*k_arr[i]*(r/2-r0/2)**2
#         force_divr_arr[j] = force_divr_arr[j] - 2*0.5*k_arr[i]*(r/2-r0/2)/r

#         # #add the energy from the angle
#         # for l,theta in enumerate(theta_arr):
#         #     # print(theta)
#         #     # print(1/2*angle_k*(theta-math.pi)**2)
#         #     bond_angle_eng_arr[j][l] = bond_eng_arr[j] + 1/2*angle_k*(theta-math.pi)**2

#     # #Find the point at which the force is 0
#     # eps = 0.1
#     # equi_r = r_arr[np.where(np.logical_and(force_divr_arr < eps,force_divr_arr > -eps))]
#     # print(equi_r)
#     # exit()

#     # print(r_arr)
#     # print(bond_eng_arr)
#     ax[0].plot(r_arr,bond_eng_arr,linewidth=2,label=rf'Bond Energy')
#     ax[1].plot(r_arr,force_divr_arr,linewidth=2,label=rf'Force/r')

#     # ------------------ Plot Formatting -----------------------
#     ax[0].set_xlabel(rf'r [$\sigma$]')
#     ax[0].set_ylabel(rf'Bond Energy [$\epsilon$]')
#     ax[0].set_title(rf'Double Well Potential for {name_arr[i]}',wrap=True,fontsize=14)
#     # ax[0].legend(loc='best',fontsize=20,columnspacing=0.1,frameon=False,ncol=1,
#     #                     labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
#     ax[0].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
#                                 bottom=True,top=True,left=True,right=True)
#     # ax.set_xticks(np.arange(min(time), max(time)*1.01, s*dt))
#     # ax.set_yticks(np.arange(min(time), max(time)*1.01, s*dt))
#     ax[0].set_xlim([0.5,2])
#     ax[0].set_ylim([0,10])
#     ax[0].grid()
#     # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#                                         #Scientific format for y axis labels
#     # ------------------ Plot Formatting -----------------------
#     ax[1].set_xlabel(rf'r [$\sigma$]')
#     ax[1].set_title(rf'Double Well Potential for {name_arr[i]}',wrap=True,fontsize=14)
#     ax[1].set_ylabel(rf'Bond Force divided by r [$\epsilon/\sigma^2$]')
#     # ax[1].legend(loc='best',fontsize=20,columnspacing=0.1,frameon=False,ncol=1,
#     #                     labelspacing=0.1,handletextpad=0.5,handlelength=0.6)
#     ax[1].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False,
#                                 bottom=True,top=True,left=True,right=True)
#     ax[1].set_xlim([0.5,2])
#     ax[1].set_ylim([-30,30])
#     ax[1].grid()

#     # ----------------- Plot display and saving ---------------
#     # if not (os.path.exists('./plots')):
#     #     os.makedirs('./plots')

# plt.tight_layout()
# # plot_file_name = plot_title.replace(' ','_')
# # plt.savefig('./plots/%s.png'%(plot_file_name))
# # plt.savefig('./plots/%s.pdf'%(plot_file_name))
# plt.show()

# # #plot the colormap energy -----------------------------
# # fig, ax = plt.subplots(1,1,figsize=(8.6*scale*cm,8.6*scale*cm),
# #             sharex=False,sharey=False)

# # X, Y = np.meshgrid(r_arr, theta_arr)
# # Z = np.log(((V_max-c_half)/(b_sq*b_sq))*((X-a/2)**2-b_sq)**2 + c_half/b*(X-a/2) + c_half + 1/2*k*(X-r0_arr[i])**2 + 1/2*angle_k*(Y-math.pi)**2)
# # contour = ax.contourf(X,Y,Z,cmap="Blues",levels = 20)
# # cbar = fig.colorbar(contour)
# # cbar.ax.set_ylabel(rf'$ln(Total Energy)$')
# # contourlines = ax.contour(contour, levels=contour.levels[::2], colors='r')
# # # ax[0].scatter(r_arr,theta_arr,c=bond_angle_eng_arr.flatten(),cmap="Blues",linewidth=2,label=rf'Bond Energy')
# # # ax.colorbar()
# # ax.set_xlim([0.5,2])
# # ax.set_ylim([math.pi/2,3/2*math.pi])
# # ax.set_xlabel(rf'r [$\sigma$]')
# # ax.set_ylabel(rf'Bond Angle [rad]')
# # plt.tight_layout()
# # plt.show()
'''
