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
N = 10**5
alpha_a = 2.0
alpha_g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
dim = [1,2,3,4]
N_s = 40

for j in range(len(dim)):
    for k in range(len(alpha_g)):
        FunctionsFile.JsonGenerate(N, alpha_a, alpha_g[k], dim[j])
        FunctionsFile.multithread_pc(N, N_s)

FunctionsFile.permission_run(N)