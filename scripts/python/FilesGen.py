import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = [20000, 40000, 80000, 160000, 320000]
alpha_g_f = 2.0
alpha_a_v = [0.0, 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

alpha_g_v = [0.0, 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
alpha_a_f = 2.0

dim = [1,2,3,4]
N_s = [80, 40, 20, 10, 5]

for n in range(len(N)):
	for i in range(len(dim)):
		for aa in range(len(alpha_a_v)):
			FunctionsFile.JsonGenerate(N[n], alpha_a_v[aa], alpha_g_f, dim[i])
			FunctionsFile.ScriptGenerate(N[n], alpha_a_v[aa], alpha_g_f, dim[i], N_s[n])
			
			FunctionsFile.JsonGenerate(N[n], alpha_a_f, alpha_g_v[aa], dim[i])
			FunctionsFile.ScriptGenerate(N[n], alpha_a_f, alpha_g_v[aa], dim[i], N_s[n])
		                
FunctionsFile.text_terminal()
