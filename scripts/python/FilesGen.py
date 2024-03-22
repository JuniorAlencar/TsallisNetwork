import src.GenerateFunctions as FunctionsFile
import numpy as np

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

dim = [1,2,3,4]
N = 80000
alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
alpha_g = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
dim = [1 ,2, 3, 4]
N_s = 30

for d in dim:
    for i in alpha_a:
        for j in alpha_g:
            FunctionsFile.JsonGenerate(N, i, j, d)
            FunctionsFile.ScriptGenerate(N, i, j, d, N_s)

FunctionsFile.text_terminal()
