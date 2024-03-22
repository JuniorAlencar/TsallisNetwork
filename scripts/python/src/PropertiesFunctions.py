import numpy as np
import pandas as pd
import os
import gzip
import glob
import networkx as nx
from decimal import Decimal, getcontext

# Return: dataframe with all samples
def all_properties_dataframe(N,dim,alpha_a,alpha_g):
    # Directory with all samples
    path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}"
    # dataframe with all samples
    new_file = "/properties_set.txt"
    
    # Check if directory exist
    conditional_ = os.path.exists(path_d)
    if(conditional_ == True):
        pass
    else:
        return "data doesn't exist, run code in c++ to gen data"
    
    # Check if file exist
    check_file = os.path.isfile(path_d+new_file)
    
    # Open all files path in directory .csv
    all_files = glob.glob(os.path.join(path_d,"*.csv"))
    # If file exist, open
    if(check_file == True):
        df = pd.read_csv(path_d+new_file,sep=" ")
        data = df.iloc[:,1:]
        #filter_list to check if files are in dataframe
        filter_list = str(data["#cod_file"].values) 
        num_samples = len(data)
        for file in all_files:
            # Check if file are in dataframe
            conditional = os.path.basename(file)[5:-4] in filter_list
            # Make nothing if True conditional
            if(conditional==True):
                pass
            # Add new elements in dataframe
            else:
                new_data = pd.read_csv(file)
                df.loc[num_samples,"#short_path"] = new_data["#mean shortest path"].values[0]
                df.loc[num_samples,"#diamater"] = new_data["# diamater"].values[0]
                df.loc[num_samples,"#ass_coeff"] = new_data["#assortativity coefficient"].values[0]
                df.loc[num_samples,"#cod_file"] = os.path.basename(file)[5:-4]
                num_samples += 1
        # Save new dataframe update
        df.to_csv(path_d+new_file,sep=' ',index=False)

    # Else, create it
    else:
        df = pd.DataFrame(columns=["#short_path", "#diamater", "#ass_coeff", "#cod_file"])
        i = 0
        # Open all files path in directory .csv
        for file in all_files:
            train = pd.read_csv(file)
            df.loc[i,"#short_path"] = train["#mean shortest path"].values[0]
            df.loc[i,"#diamater"] = train["# diamater"].values[0]
            df.loc[i,"#ass_coeff"] = train["#assortativity coefficient"].values[0]
            df.loc[i,"#cod_file"] = os.path.basename(file)[5:-4]
            i += 1
        df.to_csv(path_d+new_file,sep=' ',index=False)

# List all pair of (alpha_a,alpha_g) folders in (N,dim) folder
def list_all_folders(N,dim):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = [(lst_folders[i][8:11],lst_folders[i][20:]) for i in range(len(lst_folders))]
    
    return set_parms

def list_all_folders_for_alpha_fixed(N,dim,alpha_g_variable):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = [(lst_folders[i][8:11],lst_folders[i][20:]) for i in range(len(lst_folders))]

    if(alpha_g_variable==True):
        alpha_a = 1.0
        alpha_g_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][0]==str(alpha_a)):
                alpha_g_V.append(float(set_parms[i][1]))
        return alpha_g_V
    else:
        alpha_g = 2.0
        alpha_a_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][1]==str(alpha_g)):
                alpha_a_V.append(float(set_parms[i][0]))
        return alpha_a_V



# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def create_all_properties_file(N,dim,alpha_a,alpha_g,alpha_g_variable):
    if(alpha_g_variable==True):
        df_all = pd.DataFrame(columns=["#alpha_g","#short_mean","#diamater_mean","#ass_coeff_mean","#short_err","#diamater_err","#ass_coeff_err","#n_samples"])
        j = 0

        for i in alpha_g:
            if(i==str(0.0)):
                pass
            else:
                file = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{i}/properties_set.txt"
                df = pd.read_csv(file, sep=' ')
                df_all.loc[j,"#alpha_g"] = i
                df_all.loc[j,"#short_mean"] = df["#short_path"].mean()
                df_all.loc[j,"#diamater_mean"] = df["#diamater"].mean()
                df_all.loc[j,"#ass_coeff_mean"] = df["#ass_coeff"].mean()
                df_all.loc[j,"#short_err"] = df["#short_path"].sem()
                df_all.loc[j,"#diamater_err"] = df["#diamater"].sem()
                df_all.loc[j,"#ass_coeff_err"] = df["#ass_coeff"].sem()
                df_all.loc[j,"#n_samples"] = len(df["#diamater"])

            j += 1
        df_all.to_csv(f"../../data/N_{N}/dim_{dim}/properties_all_alpha_a_{alpha_a}.txt",sep = ' ',index=False,mode="w")
    else:       
        df_all = pd.DataFrame(columns=["#alpha_a","#short_mean","#diamater_mean","#ass_coeff_mean","#short_err","#diamater_err","#ass_coeff_err","#n_samples"])
        j = 0

        for i in alpha_a:
            file = f"../../data/N_{N}/dim_{dim}/alpha_a_{i}_alpha_g_{alpha_g}/properties_set.txt"
            df = pd.read_csv(file, sep=' ')
            df_all.loc[j,"#alpha_a"] = i
            df_all.loc[j,"#short_mean"] = df["#short_path"].mean()
            df_all.loc[j,"#diamater_mean"] = df["#diamater"].mean()
            df_all.loc[j,"#ass_coeff_mean"] = df["#ass_coeff"].mean()
            df_all.loc[j,"#short_err"] = df["#short_path"].sem()
            df_all.loc[j,"#diamater_err"] = df["#diamater"].sem()
            df_all.loc[j,"#ass_coeff_err"] = df["#ass_coeff"].sem()
            df_all.loc[j,"#n_samples"] = len(df["#diamater"])

            j += 1
        df_all.to_csv(f"../../data/N_{N}/dim_{dim}/properties_all_alpha_g_{alpha_g}.txt",sep = ' ',index=False,mode="w")

def list_N_folders():
    directory = f"../../data/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    dif = lst_folders[0]
    diff_N = [i[2:] for i in dif]
    diff_N.sort(key=int)
    return diff_N

# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def shortest_path_dim_N(N,dim,alpha_g_variable):
    #directory = f"../../data/"
    #lst_folders = []
    #for root, dirs, files in os.walk(directory):
        #lst_folders.append(dirs)
    #dif = lst_folders[0]
    #diff_N = [i[2:] for i in dif]
    #diff_N.sort(key=int)
    diff_head_N = ["N_" + i for i in N]    
    diff_head_err = ["N_"+ i + "_err" for i in N]

    if(alpha_g_variable==True):
        diff_head_N = ["#alpha_g"] + diff_head_N
        header = diff_head_N + diff_head_err
        df_all = pd.DataFrame(columns=header)
        alpha_a = 1.0
        j = 0
        for i in N:
            if(i==str(0.0)):
                pass
            else:
                file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_a_1.0.txt"
                df = pd.read_csv(file, sep=' ')
                df_all["#alpha_g"] = df["#alpha_g"]
                df_all[f"{i}"] = df["#short_path"]
                df_all["#short_err"] = df["#short_err"]
            j += 1
        df_all.to_csv(f"../../data/short_N_dim_{dim}_{alpha_a}.txt",sep = ' ',index=False,mode="w")
    else:       
        diff_head_N = ["#alpha_a"] + diff_head_N
        header = diff_head_N + diff_head_err
        df_all = pd.DataFrame(columns=header)
        j = 0

        for i in N:
            file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_g_2.0.txt"
            df = pd.read_csv(file, sep=' ')
            df_all["#alpha_a"] = df["#alpha_a"]
            df_all[f"{i}"] = df["#short_path"]
            df_all["#short_err"] = df["#short_err"]

            j += 1
        df_all.to_csv(f"../../data/short_N_dim_{dim}_{alpha_g}.txt",sep = ' ',index=False,mode="w")

def linear_regression(X,Y,Error):
    # Calculate weighted means
    weighted_x_mean = np.sum(X / Error) / np.sum(1 / Error)
    weighted_y_mean = np.sum(Y / Error) / np.sum(1 / Error)

    # Calculate weighted covariance and variance
    weighted_covar = np.sum(X * Y / Error) / np.sum(1 / Error) - weighted_x_mean * weighted_y_mean
    weighted_x_var = np.sum(X ** 2 / Error) / np.sum(1 / Error) - weighted_x_mean ** 2

    # Calculate the slope (m) and intercept (b) of the linear regression line
    slope = weighted_covar / weighted_x_var
    intercept = weighted_y_mean - slope * weighted_x_mean
    
    y_pred = [slope*i+intercept for i in X]
#    return slope, intercept
    return y_pred

def kappa(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 0.3
    else:
        return -1.15*np.exp(1-ration)+1.45
    
def Lambda(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 1/0.3
    else:
        return 1/(-1.15*np.exp(1-ration)+1.45)

def q(alpha_a,d):
    ration = alpha_a/d
    if(0 <= ration <= 1):
        return 4/3
    else:
        return (1/3)*np.exp(1-ration)+1
    

def r_properties_dataframe(N, dim, alpha_a, alpha_g):
    if(alpha_g==str(0.0)):
        pass
    else:
        # Directory with all samples
        path_d = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/"
        # dataframe with all samples
        new_file = "/properties_set_r.txt"

        # Check if directory exist
        conditional_ = os.path.exists(path_d)
        if(conditional_ == True):
            pass
        else:
            print("data doesn't exist, run code in c++ to gen data")

        # Check if file exist
        check_file = os.path.isfile(path_d+new_file)

        # Open all files path in directory .csv
        all_files = glob.glob(os.path.join(path_d,"*.gz"))
        # If file exist, open
        if(check_file == True):
            df = pd.read_csv(path_d+new_file,sep=" ")
            #filter_list to check if files are in dataframe
            filter_list = str(df["#cod_file"].values) 
            num_samples = len(df)
            for file in all_files:
                # Check if file are in dataframe
                conditional = os.path.basename(file)[4:-7] in filter_list
                # Make nothing if True conditional
                if(conditional==True):
                    pass
                # Add new elements in dataframe
                else:
                    # load node properties
                    node = {"id": [],
                    "position":[],
                    "degree": []}
                    
                    # load edge properties
                    edge = {"connections": [],
                            "distance": []}
                    
                    with gzip.open(file) as file_in:
                        String = file_in.readlines()
                        Lines = [i.decode('utf-8') for i in String]
                        for i in range(len(Lines)):
                            if(Lines[i]=='node\n'):
                                node["id"].append(int(Lines[i+2][4:-2]))
                                node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
                                if(Lines[i+9][0]=='q'):
                                    node["degree"].append(int(Lines[i+10][7:-1]))
                                else:
                                    node["degree"].append(int(Lines[i+9][7:-1]))
                            elif(Lines[i]=="edge\n"):
                                edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
                                edge["distance"].append(float(Lines[i+4][9:-1]))
                    
                    D = np.array(node["degree"])
                    getcontext().prec = 50  # Set precision to 50 decimal places
                    Ter_1 = Decimal(int(sum(D)))
                    Ter_3 = Decimal(int(np.dot(D,D)))
                    Ter_4 = Decimal(int(sum(D**3)))
                    
                    G = nx.from_edgelist(edge["connections"])
                    Ter_2 = 0
                    
                    for j in G.edges():
                        d_s = G.degree[j[0]]
                        d_t = G.degree[j[1]]
                        Ter_2 += d_s*d_t 
                    
                    Ter_2 = Decimal(Ter_2)
                    
                    getcontext().prec = 10  # Set precision to 50 decimal places
                    
                    r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
                    df.loc[num_samples,"#ass_coeff"] = r
                    df.loc[num_samples,"#cod_file"] = os.path.basename(file)[4:-7]
                    num_samples += 1
            # Save new dataframe update
            df["#cod_file"] = df["#cod_file"].astype(int)
            df.to_csv(path_d+new_file,sep=' ',index=False)

        # Else, create it
        else:
            ass_coeff = []
            cod_file = []
            # Open all files path in directory .csv
            
            for file in all_files:
                node = {"id": [],
                    "position":[],
                    "degree": []}
                edge = {"connections": [],
                        "distance": []}
                with gzip.open(file) as file_in:
                    String = file_in.readlines()
                    Lines = [i.decode('utf-8') for i in String]
                    for i in range(len(Lines)):
                        if(Lines[i]=='node\n'):
                            node["id"].append(int(Lines[i+2][4:-2]))
                            node["position"].append([float(Lines[i+6][2:-1]),float(Lines[i+7][2:-1]),float(Lines[i+8][2:-1])])
                            if(Lines[i+9][0]=='q'):
                                node["degree"].append(int(Lines[i+10][7:-1]))
                            else:
                                node["degree"].append(int(Lines[i+9][7:-1]))
                        elif(Lines[i]=="edge\n"):
                            edge["connections"].append([int(Lines[i+2][8:-2]),int(Lines[i+3][8:-2])])
                            edge["distance"].append(float(Lines[i+4][9:-1]))
                D = np.array(node["degree"])
                getcontext().prec = 50  # Set precision to 50 decimal places
                Ter_1 = Decimal(int(sum(D)))
                Ter_3 = Decimal(int(np.dot(D,D)))
                Ter_4 = Decimal(int(sum(D**3)))
                G = nx.from_edgelist(edge["connections"])
                Ter_2 = 0
                for j in G.edges():
                    d_s = G.degree[j[0]]
                    d_t = G.degree[j[1]]
                    Ter_2 += d_s*d_t 
                Ter_2 = Decimal(Ter_2)
                getcontext().prec = 10  # Set precision to 50 decimal places
                r = Decimal((Ter_1*Ter_2-Ter_3*2)/(Ter_1*Ter_4-Ter_3**2))
                ass_coeff.append(r)
                cod_file.append(os.path.basename(file)[4:-7])
            df = pd.DataFrame(data={"#ass_coeff":ass_coeff,"#cod_file":cod_file})
            df.to_csv(path_d+new_file,sep=' ',index=False)
