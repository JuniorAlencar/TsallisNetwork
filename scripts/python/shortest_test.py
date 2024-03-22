import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
import time
from collections import Counter
from scipy.stats import binned_statistic
from IPython.display import clear_output
from collections import OrderedDict
import networkx as nx
import csv

#from collections import Counter
def PlotGraph(N,dim,alpha_a,alpha_g,file, layout):
    path = f'../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/connections/'
    File = f"connections_{file}.csv.gz"
    connection = pd.read_csv(path+File)
    con = [(connection["#Node1"].values[i],connection["#Node2"].values[i]) for i in range(len(connection["#Node1"]))]
    g=nx.from_edgelist(con)
    if(layout=="circular"):
        nx.draw_circular(g, with_labels = False)
        plt.savefig("circular.pdf")
        plt.clf()
    elif(layout=="planar"):
        nx.draw_planar(g, with_labels = False)
        plt.savefig("planar.pdf")
        plt.clf()
    elif(layout=="spectral"):
        nx.draw_spectral(g,with_labels = False)
        plt.savefig("spectral.pdf")
        plt.clf()
    elif(layout=="spring"):
        nx.draw_spring(g, with_labels = False, node_color="red")
        plt.savefig("spring.pdf")
        plt.clf()

PlotGraph(100,2,2.0,2.0,650547248,"spring")


# def shortest(start,end):
#     pathOriginal = '../../data/N_5000/dim_1/alpha_a_0.0_alpha_g_2.0/'
#     #all_files = glob.glob(os.path.join(pathOriginal,"*.csv"))
#     connection = pd.read_csv('../../data/N_5000/dim_1/alpha_a_0.0_alpha_g_2.0/connections/connections_16503154.csv.gz')
#     con = [(connection["#Node1"].values[i],connection["#Node2"].values[i]) for i in range(len(connection["#Node1"]))]
#     G=nx.from_edgelist(con)
#     ter_0 = 41423863
#     m = len(con)+1
#     f = open('data.csv', 'w')    
#     for i in range(start,end):
#         writer = csv.writer(f)
#         start = time.time()
#         count = 0
#         if(i==1181):
#             arroz=2297
#         else:
#             arroz=0
#         for j in range(arroz,m):
#             if(j>i):
#                     shortest = len([p for p in nx.all_shortest_paths(G, source=i, target=j,method='BFS')][0]) - 1
#                     ter_0 += shortest
#                     # Saving i,j index, and ter_0 value
#                     df = pd.DataFrame(data={"i":[i],"j":[j],"ter_0":[ter_0]}) 
#                     df.to_csv("shortest_backup.csv",mode="w+",index=False)
#                     count += 1
#         end = time.time()
#         data = [i,end-start,count]
#         writer.writerow(data)
#         print(f"i={i},time={end-start},num_elements={count}")               
#     shortest_path = ter_0 / ((m/2)*(m-1))
#     f.close()
#     return shortest_path, ter_0

# short, t0 = shortest(1181,5000)
# print(short)
