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

df = pd.read_csv("rest_parms.txt", delimiter = ' ')
N_s = 100
alpha_a = [round(i,2) for i in df["alpha_a"].values]
alpha_g = [round(i,2) for i in df["alpha_g"].values]
dim = [i for i in df["dim"].values]
interval = range(int(len(alpha_a)/2),int(len(alpha_a)))

for i in interval:
    FunctionsFile.JsonGenerate(N, alpha_a[i], alpha_g[i], dim[i])
    FunctionsFile.multithread_pc(N, N_s)

FunctionsFile.permission_run(N)
