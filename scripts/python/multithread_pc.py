import src.MultithreadPC_Functions as FunctionsFile
import numpy as np
import glob
import os

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







dim = [1,2,3,4]
N = 40000
alpha_g = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
N_s = 15

for d in dim:
    for a in alpha_a:
        for g in alpha_g:
            FunctionsFile.JsonGenerate(N, a, g, d)
            FunctionsFile.multithread_pc(N, N_s)
			
FunctionsFile.permission_run(N)
