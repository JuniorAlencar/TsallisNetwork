import src.MultithreadPC_Functions as FunctionsFile
import numpy as np
import glob
import os
import pandas as pd

#multithread_pc(N,num_samples)
#N: number of nodes;
# return .sh file to run parallel in PC in scripts folder
#--------------------------------------------------------
#JsonGenerate(N, alpha_a,alpha_g,dim)
#multithread_pc(N,NumSamples)
#alpha_a: parameter to control preferential attrachment
#alpha_g: parameter to control random power law
#dim: dimension
#N: Number of nodes;
#return: set of .json file with above parameters 

#N = [10000, 20000, 40000, 80000, 160000, 320000]
#N_s = [300, 175, 120, 42, 55, 10]
#N = [5000, 10000, 20000, 40000, 80000, 160000, 320000]
N = 5000
alpha_g_f = 2.0
alpha_a_v = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

alpha_g_v = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
alpha_a_f = 2.0
dim = 1
#dim = [1,2,3,4]
N_ = [20000, 40000, 320000]

N_s = 20000
#N_s = [10000, 5000, 1000, 500, 200, 20, 10]
#for n in range(len(N)):
	#for i in range(len(dim)):
#		for aa in range(len(alpha_a_v)):
			#FunctionsFile.JsonGenerate(N[n], alpha_a_f, alpha_g_v[aa], dim[i])
#			FunctionsFile.JsonGenerate(N[n], alpha_a_v[aa], alpha_g_f, dim[i])
Dim = [1, 2, 3, 4]	       
N_s_ = [30, 10, 5]
#for n in range(len(N)):
	#FunctionsFile.multithread_pc(N[n], N_s[n])
#	FunctionsFile.permission_run(N[n])
for aa in range(len(alpha_a_v)):
	FunctionsFile.JsonGenerate(N, alpha_a_f, alpha_g_v[aa], dim)
	FunctionsFile.JsonGenerate(N, alpha_a_v[aa], alpha_g_f, dim)

FunctionsFile.multithread_pc(N, N_s)
FunctionsFile.permission_run(N)

for n in range(3):
	for d in Dim:
		for aa in range(len(alpha_a_v)):
			FunctionsFile.JsonGenerate(N_[n], alpha_a_f, alpha_g_v[aa], d)
			FunctionsFile.JsonGenerate(N_[n], alpha_a_v[aa], alpha_g_f, d)

	FunctionsFile.multithread_pc(N_[n], N_s_[n])
	FunctionsFile.permission_run(N_[n])


