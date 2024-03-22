import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from os import path

def alpha_a(gamma,d):
    alphaA = np.zeros(len(gamma))
    for i in range(len(gamma)):
        if(gamma[i] == 1/0.3):
            alphaA[i] = d-0.4
        else:
            alphaA[i] = d*(1-np.log((gamma[i]*1.45-1)/(1.15*gamma[i])))
    return alphaA

# Function for create files json with parameters network.
def json_generate(gamma, d, N):    
    
    if not os.path.exists("../params/"):
        #path = os.path.join("../", "params")
        os.makedirs("../params/")
    
    #path = os.path.join("../", "params")  # Create path with name directory
    alpha = alpha_a(gamma,d)
    #os.mkdir(path) # Create directory ../params/
    
    for i in range(len(gamma)):
        arroz = alpha[i]
        name = "g" + str(round(gamma[i],4)) + "_" + "d" + str(d) + ".json"
        dictionary = {
        "comment": "use seed = -1 for random seed",
        "num_vertices": N,
        "alpha_a": round(arroz,6),
        "alpha_g": 2,
        "r_min": 1,
        "r_max": 10000000,
        "dim": d,
        "seed": -1
        }
            
        json_object = json.dumps(dictionary, indent=8)
        with open("../params/" + name, "w") as outfile:
            outfile.write(json_object)
            outfile.close()


# Example params
gamma = np.linspace(1/0.3,1/1.45+0.001,4)   # Gamma Values: (0.7, 1 , 10, 100); 0.68 <= gamma <= 3.3;
d = [1,2,3,4]            # Dimensions values: (1, 2, 3, 4);
N = 10**5                # Number of nodes: 100.000;
json_generate(gamma,d[0],N)
json_generate(gamma,d[1],N)
json_generate(gamma,d[2],N)
json_generate(gamma,d[3],N)



# #print(gamma_test(R_a))
# plt.plot(gamma,alpha_a(gamma,d[0]),'o',label='d=1')
# plt.plot(gamma,alpha_a(gamma,d[1]),'o',label='d=2')
# plt.plot(gamma,alpha_a(gamma,d[2]),'o',label='d=3')
# plt.plot(gamma,alpha_a(gamma,d[3]),'o',label='d=4')
# plt.xlabel(r"$\gamma$",size=20)
# plt.ylabel(r"$\alpha_A$",size=20)
# #plt.xlim([0,9])
# plt.legend()
# plt.show()


# def eta_tes(alpha_a,d):
#     eta = np.zeros(len(alpha_a))
#     for i in range(len(alpha_a)):
#         if(alpha_a[i]/d >= 0 and alpha_a[i]/d <=1):
#             eta[i] = 0.3
#         else:
#             eta[i] = -1.15*np.exp(1-alpha_a[i]/d) + 1.45
#     return eta

# def gamma_test(alpha_a,d):
#     gamma = np.zeros(len(alpha_a))
#     for i in range(len(alpha_a)):
#         if(alpha_a[i]/d >= 0 and alpha_a[i]/d <=1):
#             gamma[i] = 1/0.3
#         elif(alpha_a[i]>1):
#             gamma[i] = 1/(-1.15*np.exp(1-alpha_a[i]/d) + 1.45)
#     return gamma





