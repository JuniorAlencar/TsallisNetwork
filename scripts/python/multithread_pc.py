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

N = 5000
N_s = 100
df = pd.read_csv("new_data.csv", delimiter = ',')
alpha_a = [i for i in df[df["dim"]==4]['alpha_a'].values]
alpha_g = [i for i in df[df["dim"]==4]['alpha_g'].values]
dim = [i for i in df[df["dim"]==4]['dim'].values]

for i in range(len(dim)):
    FunctionsFile.JsonGenerate(N, alpha_a[i], alpha_g[i], 4)
    FunctionsFile.multithread_pc(N, N_s)

FunctionsFile.permission_run(N)

