import src.GenerateFunctions as FunctionsFile
import numpy as np
import pandas as pd

# FunctionsFile defined in folder ./src
#ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples) -> generate files .sh to run in cluster
#JsonGenerate(N, alpha_a, alpha_g, dim)              -> generate files .json for entry in script
#text_terminal()                                    -> return .txt with text to run codes in cluster
#------------------------------------------------------------------------------------------

N = 10**5
alpha_g = 2.0
alpha_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
dim = [1,2,3,4]
N_s = 120

for i in range(len(dim)):
    for aa in alpha_a:
        FunctionsFile.JsonGenerate(N, aa, alpha_g, dim[i])
        FunctionsFile.ScriptGenerate(N, aa, alpha_g, dim[i], N_s)
                        
FunctionsFile.text_terminal()
