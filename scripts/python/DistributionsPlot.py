import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.DistributionsFunctions import q, eta, ln_q
# import matplotlib.pyplot as plt
# import os


# Functions defined in folder ./src
#degree(N, dim, alpha_a, alpha_g)              --> Return concatenate list of all samples degree
#log_binning(degree, log_min, log_max, N_bins) --> Return k_log_binning,pk_log_binning (log_min/max = expoent in base 10)
#distribution(degree)                          --> Return k, pk em linear binning
#cumulative_distribution(distribution)         --> Return p_cum
#------------------------------------------------------------------------------------------------------------------------>

# # Loading data ------------->
N = 10**5
dim = [1,2,3,4]

alpha_a1D = [0.0,1.0,1.5,2.0,3.0]
alpha_a2D = [0.0,2.0,2.5,3.0,5.0]
alpha_a3D = [0.0,3.0,3.5,4.0,5.0]
alpha_a4D = [0.0,4.0,5.0,6.0,8.0]
#alpha_a = 2.0
alpha_g = 2.0
# path_lin_1d = f"../../data/N_{N}/distributions/linear/N{N}_d{1}_alphaA{alpha_a1D[0]}_alphaG{2.0}.csv"
# #k_1d, lnq_1d_lin = pd.read_csv(path_lin_1d)["k"].values, ln_q(pd.read_csv(path_lin_1d.values)["pk"],q(alpha_a,1),eta(alpha_a,1))
# def f(x):
#     return 2*x
# k, ll = pd.read_csv(path_lin_1d)["k"].values, ln_q(pd.read_csv(path_lin_1d)["pk"].values,q(alpha_a1D[0],1),eta(alpha_a1D[0] ,1))
# print(ll)
lnq_1d_lin = [[] for _ in range(len(alpha_a1D))]
lnq_2d_lin = [[] for _ in range(len(alpha_a1D))]
lnq_3d_lin = [[] for _ in range(len(alpha_a1D))]
lnq_4d_lin = [[] for _ in range(len(alpha_a1D))]

lnq_1d_log = [[] for _ in range(len(alpha_a1D))]
lnq_2d_log = [[] for _ in range(len(alpha_a1D))]
lnq_3d_log = [[] for _ in range(len(alpha_a1D))]
lnq_4d_log = [[] for _ in range(len(alpha_a1D))]

k_1d_lin = [[] for _ in range(len(alpha_a1D))]
k_2d_lin = [[] for _ in range(len(alpha_a1D))]
k_3d_lin = [[] for _ in range(len(alpha_a1D))]
k_4d_lin = [[] for _ in range(len(alpha_a1D))]


k_1d_log= [[] for _ in range(len(alpha_a1D))]
k_2d_log= [[] for _ in range(len(alpha_a1D))]
k_3d_log= [[] for _ in range(len(alpha_a1D))]
k_4d_log= [[] for _ in range(len(alpha_a1D))]

pk1d = [[] for _ in range(len(alpha_a1D))]
pk2d = [[] for _ in range(len(alpha_a1D))]
pk3d = [[] for _ in range(len(alpha_a1D))]
pk4d = [[] for _ in range(len(alpha_a1D))]

pklog1 = [[] for _ in range(len(alpha_a1D))]
pklog2 = [[] for _ in range(len(alpha_a1D))]
pklog3 = [[] for _ in range(len(alpha_a1D))]
pklog4 = [[] for _ in range(len(alpha_a1D))]



for i in range(len(alpha_a1D)):
    path_lin_1d = f"../../data/N_{N}/distributions/linear/N{N}_d{1}_alphaA{alpha_a1D[i]}_alphaG{2.0}.csv"
    path_log_1d = f"../../data/N_{N}/distributions/log_binning/N{N}_d{1}_alphaA{alpha_a1D[i]}_alphaG{2.0}.csv"
    
    path_lin_2d = f"../../data/N_{N}/distributions/linear/N{N}_d{2}_alphaA{alpha_a2D[i]}_alphaG{2.0}.csv"
    path_log_2d = f"../../data/N_{N}/distributions/log_binning/N{N}_d{2}_alphaA{alpha_a2D[i]}_alphaG{2.0}.csv"
    
    path_lin_3d = f"../../data/N_{N}/distributions/linear/N{N}_d{3}_alphaA{alpha_a3D[i]}_alphaG{2.0}.csv"
    path_log_3d = f"../../data/N_{N}/distributions/log_binning/N{N}_d{3}_alphaA{alpha_a3D[i]}_alphaG{2.0}.csv"
    
    path_lin_4d = f"../../data/N_{N}/distributions/linear/N{N}_d{4}_alphaA{alpha_a4D[i]}_alphaG{2.0}.csv"
    path_log_4d = f"../../data/N_{N}/distributions/log_binning/N{N}_d{4}_alphaA{alpha_a4D[i]}_alphaG{2.0}.csv"
    
    #k_1d[i], lnq_1d_lin[i] = pd.read_csv(path_lin_1d)["k"].values, ln_q(pd.read_csv(path_lin_1d.values)["pk"],q(alpha_a1D[i],1),eta(alpha_a1D[i],1))
    k_1d_lin[i], pk1d[i] = pd.read_csv(path_lin_1d)["k"].values, pd.read_csv(path_lin_1d)["pk"].values
    k_2d_lin[i], pk2d[i] = pd.read_csv(path_lin_2d)["k"].values, pd.read_csv(path_lin_2d)["pk"].values
    k_3d_lin[i], pk3d[i] = pd.read_csv(path_lin_3d)["k"].values, pd.read_csv(path_lin_3d)["pk"].values
    k_4d_lin[i], pk4d[i] = pd.read_csv(path_lin_4d)["k"].values, pd.read_csv(path_lin_4d)["pk"].values
    lnq_1d_lin[i] = ln_q(k_1d_lin[i],pk1d[i],q(alpha_a1D[i],1),eta(alpha_a1D[i],1))
    lnq_2d_lin[i] = ln_q(k_2d_lin[i],pk2d[i],q(alpha_a2D[i],2),eta(alpha_a2D[i],2))
    lnq_3d_lin[i] = ln_q(k_3d_lin[i],pk3d[i],q(alpha_a3D[i],3),eta(alpha_a3D[i],3))
    lnq_4d_lin[i] = ln_q(k_4d_lin[i],pk4d[i],q(alpha_a4D[i],4),eta(alpha_a4D[i],4))
    
    #k_4d_lin[i], pk4d = pd.read_csv(path_lin_4d)["k"].values, ln_q(pd.read_csv(path_lin_4d)["pk"].values,q(alpha_a4D[i],1),eta(alpha_a4D[i] ,1))
    #k_1d_log[i], lnq_1d_log[i] = pd.read_csv(path_log_1d)["k"],ln_q(pd.read_csv(path_log_1d)["pk"],q(alpha_a1D[i],1),eta(alpha_a1D[i],1))
    k_1d_log[i], pklog1[i] = pd.read_csv(path_log_1d)["k"].values, pd.read_csv(path_log_1d)["pk"].values
    k_2d_log[i], pklog2[i] = pd.read_csv(path_log_2d)["k"].values, pd.read_csv(path_log_1d)["pk"].values
    k_3d_log[i], pklog3[i] = pd.read_csv(path_log_3d)["k"].values, pd.read_csv(path_log_1d)["pk"].values
    k_4d_log[i], pklog4[i] = pd.read_csv(path_log_4d)["k"].values, pd.read_csv(path_log_1d)["pk"].values

    lnq_1d_log[i] = ln_q(k_1d_log[i],pk1d[i],q(alpha_a1D[i],1),eta(alpha_a1D[i],1))
    lnq_2d_log[i] = ln_q(k_2d_log[i],pk2d[i],q(alpha_a2D[i],2),eta(alpha_a2D[i],2))
    lnq_3d_log[i] = ln_q(k_3d_log[i],pk3d[i],q(alpha_a3D[i],3),eta(alpha_a3D[i],3))
    lnq_4d_log[i] = ln_q(k_4d_log[i],pk4d[i],q(alpha_a4D[i],4),eta(alpha_a4D[i],4))

color = ["red","green","yellow","blue","cyan"]
label1d = [rf"$\alpha_a$ = {i}" for i in alpha_a1D]
label2d = [rf"$\alpha_a$ = {i}" for i in alpha_a2D]
label3d = [rf"$\alpha_a$ = {i}" for i in alpha_a3D]
label4d = [rf"$\alpha_a$ = {i}" for i in alpha_a4D]

fig, ax = plt.subplots(2, 2,figsize=(15,10))
fig.suptitle(r'$\alpha_a=2.0$', fontsize=20)
for i in range(len(alpha_a1D)):
    # ax[0, 0].plot(k_1d_lin[i],lnq_1d_lin[i],'o',color=color[i],label=label1d[i])
    # ax[0, 1].plot(k_2d_lin[i],lnq_2d_lin[i],'o',color=color[i],label=label2d[i])
    # ax[1, 0].plot(k_3d_lin[i],lnq_3d_lin[i],'o',color=color[i],label=label3d[i])
    # ax[1, 1].plot(k_4d_lin[i],lnq_4d_lin[i],'o',color=color[i],label=label4d[i])
    ax[0, 0].plot(k_1d_log[i],pklog1[i],'o',color=color[i],label=label1d[i])
    ax[0, 1].plot(k_2d_lin[i],pk2d[i],'o',color=color[i],label=label2d[i])
    ax[1, 0].plot(k_3d_lin[i],pk3d[i],'o',color=color[i],label=label3d[i])
    ax[1, 1].plot(k_4d_lin[i],pk4d[i],'o',color=color[i],label=label4d[i])

#ax[0, 0].plot(x1[0:s1], linear_regression(x1[:s1],a_my[0],b_my[0]),'-',color='black')
#ax[0, 1].plot(x2[0:s2], linear_regression(x2[:s2],a_my[1],b_my[1]),'-',color='black')
#ax[1, 0].plot(x3[0:s3], linear_regression(x3[:s3],a_my[2],b_my[2]),'-',color='black')
#ax[1, 1].plot(x4[0:s4], linear_regression(x4[:s4],a_my[3],b_my[3]),'-',color='black')


# ax[0, 0].set_ylim([-100,0])
# ax[0, 1].set_ylim([-30,0])
# ax[1, 0].set_ylim([-10,0])
# ax[1, 1].set_ylim([-5,0])
for i in range(2):
    for j in range(2):
        #ax[i, j].set_xlim([0,100])
        #ax[i, j].set_ylim([-100,0])
        ax[i, j].set_xlim([1,10**4])
        ax[i, j].set_ylim([10**(-6.5),1])
        ax[i, j].set_yscale('log')
        ax[i, j].set_xscale('log')
        ax[i, j].legend()
        ax[i, 0].set_ylabel(r"$\ln_q[P(k)/P(0)]$",size=15)
        ax[1, j].set_xlabel(r"$k$",size=15)
plt.show()

# fig, ax = plt.subplots(2, 2,figsize=(15,10))
# fig.suptitle(r'$\alpha_a=2.0$', fontsize=20)
# for i in range(len(alpha_a1D)):
#     ax[0, 0].plot(k_1d_log[i],lnq_1d_log[i],'o',color=color[i],label=label1d[i])
#     ax[0, 1].plot(k_2d_log[i],lnq_2d_log[i],'o',color=color[i],label=label2d[i])
#     ax[1, 0].plot(k_3d_log[i],lnq_3d_log[i],'o',color=color[i],label=label3d[i])
#     ax[1, 1].plot(k_4d_log[i],lnq_4d_log[i],'o',color=color[i],label=label4d[i])

# #ax[0, 0].plot(x1[0:s1], linear_regression(x1[:s1],a_my[0],b_my[0]),'-',color='black')
# #ax[0, 1].plot(x2[0:s2], linear_regression(x2[:s2],a_my[1],b_my[1]),'-',color='black')
# #ax[1, 0].plot(x3[0:s3], linear_regression(x3[:s3],a_my[2],b_my[2]),'-',color='black')
# #ax[1, 1].plot(x4[0:s4], linear_regression(x4[:s4],a_my[3],b_my[3]),'-',color='black')



# for i in range(2):
#     for j in range(2):
#         ax[i, j].set_xlim([0,100])
#         ax[i, j].set_ylim([-100,0])
#         ax[i, j].legend()
#         ax[i, 0].set_ylabel(r"$\ln_q[P(k)/P(0)]$",size=15)
#         ax[1, j].set_xlabel(r"$k$",size=15)
# plt.show()