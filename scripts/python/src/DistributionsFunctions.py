import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
import gzip

# N: number of nodes
# d: dimension
# alpha_a: parameter for preferential attrachment
# alpha_g: parameter for probability of distances nodes

# return(degree): list of degree with all samples

def degree(N, d,alpha_a,alpha_g):
    path_d = f"../../data/N_{N}/dim_{d}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/degree"
    all_files = glob.glob(os.path.join(path_d,"*.csv.gz"))
    li = []
    for file in all_files:
        train = pd.read_csv(file)
        li.append(train)
        #train.values.flatten() # Select just values of degree
        
    frame = pd.concat(li)
    frame = pd.concat(li, ignore_index=True)
    degree = frame["k"].values
    return degree

def distribution(degree):
    hist, bins_edge = np.histogram(degree, bins=np.arange(0.5,10**4+1.5,1), density=True)
    
    P = hist*np.diff(bins_edge)             # distribution = density*deltaK
    K = bins_edge[:-1]+bins_edge[:1]
    index_remove = []                       # load index with distribution zero
    
    for idk,elements in enumerate(P):
        if(elements==0):
            index_remove.append(idk) 
    # Removing elements in k_mean and distribution with distribution = 0 (empty box)
    p_real = np.delete(P,index_remove)      
    k_real = np.delete(K,index_remove)
    return k_real,p_real

# distribution: distribution (N_k/N)
# return(p_cum): cumulative distribution

def cumulative_distribution(distribution):
    p_cum = np.zeros(len(distribution))
    p_cum[0] = sum(distribution)
    for i in range(1,len(distribution)):
        p_cum[i] = p_cum[i-1] - distribution[i]
    return p_cum

# Logbinnig_test

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(counter_dict,bin_count):

    max_x = np.log10(max(list(counter_dict.keys())))
    max_y = np.log10(max(list(counter_dict.values())))
    max_base = max([max_x,max_y])

    min_x = np.log10(min(drop_zeros(list(counter_dict.keys()))))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    Pk = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.values()), density = True)[0] / np.histogram(list(counter_dict.keys()),bins)[0])*np.diff(bins)
    k = (np.histogram(list(counter_dict.keys()),bins,weights=list(counter_dict.keys()))[0] / np.histogram(list(counter_dict.keys()),bins)[0])

    k = [x for x in k if str(x) != 'nan']
    Pk = [x for x in Pk if str(x) != 'nan']

    return k,Pk

def q(alpha_a,d):
    ration = alpha_a/d
    if(0<=ration<=1):
        return 4/3
    else:
        return (1/3)*np.exp(1-ration)+1

def eta(alpha_a,d):
    ration = alpha_a/d
    if(0<=ration<=1):
        return 0.3
    else:
        return -1.15*np.exp(1-ration) + 1.45

def ln_q(k, pk,distribution, q, eta):
    k_values = np.zeros(len(k))
    for i in range(len(k)):
        k_values[i] = (1+(q-1)*(k/eta))**(1/(1-q))
    P0 = sum(k_values)
    return ((pk/P0)**(1-q)-1)/(1-q)

# Save all_data distributions (degree and distances)
def create_all_distributions(N,dim,alpha_a,alpha_g):
    # open folder with parameters (N,dim,alpha_a,alpha-g)
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml" 
    # folder where save dataframe with all data 
    path_degree = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/"
    
    # check if file exist
    isExist = os.path.exists(path)

    if(isExist==False):
        print("doesn't exist gml_folder")

    all_files = glob.glob(os.path.join(path,"*.gml.gz"))

    isExistDegreeAll = os.path.exists(path_degree + "degree_all.csv")
    isExistRAll = os.path.exists(path_degree + "dist_all.csv")
    # if file doesn't exist create it
    if(isExistDegreeAll==False and isExistRAll==False):
        code_dict = {"cod_files":[]} # save cod_file
        degree = {"degree":[]}       # save degree
        dist = {"dist":[]}           # save distances
 
        for file in all_files: # run for all files in folder
            file_name = os.path.basename(file)
            code_dict["cod_files"].append(file_name) #save cod (seed)
            with gzip.open(file) as file_in:
                String = file_in.readlines()
                Lines = [i.decode('utf-8') for i in String]
                for i in range(len(Lines)):
                    if(Lines[i]=='node\n'):
                        degree["degree"].append(int(Lines[i+9][7:-1]))
                    elif(Lines[i]=="edge\n"):
                        dist["dist"].append(float(Lines[i+4][9:-1]))
        # dataframe with codes
        df_c = pd.DataFrame(data=code_dict,dtype="str")
        # dataframe with degree
        df_degree = pd.DataFrame(data=degree)
        # dataframe with distances
        df_dist = pd.DataFrame(data=dist)

        df_c.to_csv(path_degree + "cod_files.csv",index=False)
        df_degree.to_csv(path_degree + "degree_all.csv",index=False)
        df_dist.to_csv(path_degree + "dist_all.csv",index=False)
    else: # if files exists, just check if need update
        df_c = pd.read_csv(path_degree + "/cod_files.csv")
        df_degree = pd.read_csv(path_degree + "/degree_all.csv")
        df_dist = pd.read_csv(path_degree + "/dist_all.csv")
        for file in all_files:
            file_name = os.path.basename(file)
            bolean_file_name = file_name in df_c["cod_files"].values
            
            # if file in cod_file, just pass
            if(bolean_file_name == True): 
                pass
            # else update cod_file, distances and degree dataframe
            else:
                lst_degree = [] # aux degree
                lst_dist = []   # aux distances
                new_degree = pd.DataFrame(columns=df_degree.columns)
                new_dist = pd.DataFrame(columns=df_dist.columns)
                
                new_c = pd.DataFrame(columns=df_c.columns)
                new_c['cod_files'] = [file_name]
                df_c = pd.concat([df_c, new_c], ignore_index=True)

                with gzip.open(file) as file_in:
                    String = file_in.readlines()
                    Lines = [i.decode('utf-8') for i in String]
                    for i in range(len(Lines)):
                        if(Lines[i]=='node\n'):
                            lst_degree.append(int(Lines[i+9][7:-1]))
                        elif(Lines[i]=="edge\n"):
                            lst_dist["dist"].append(float(Lines[i+4][9:-1]))
                new_degree['degree'] = lst_degree
                new_dist['degree'] = lst_dist
                df_d = pd.concat([df_d, new_degree], ignore_index=True)
            df_degree.to_csv(path_degree + "/degree_all.csv",index=False,mode='w')
            df_c.to_csv(path_degree + "/cod_files.csv",index=False,mode='w')
            df_dist.to_csv(path_degree + "/dist_all.csv",index=False,mode='w')