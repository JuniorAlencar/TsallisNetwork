import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

# def alpha_a_values(d):
#     list_values = []
#     multiplys = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
#     for i in range(len(multiplys)):
#         new_values = [multiplys[i]*j for j in d]
#         list_values.append(new_values)
#     return list_values

N = 5000
N_s = 100
df = pd.read_csv("new_data.csv", delimiter = ',')
alpha_a = [i for i in df[df["dim"]==3]['alpha_a'].values]
alpha_g = [i for i in df[df["dim"]==3]['alpha_g'].values]
dim = [i for i in df[df["dim"]==3]['dim'].values]

for i in range(len(dim)):
    FunctionsFile.JsonGenerate(N, alpha_a[i], alpha_g[i], 3)
    FunctionsFile.ScriptGenerate(N, alpha_a[i], alpha_g[i], 3, N_s)
                        
FunctionsFile.text_terminal()