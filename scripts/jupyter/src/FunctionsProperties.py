import numpy as np
import pandas as pd
import os
import gzip
import glob
import networkx as nx
from decimal import Decimal, getcontext
from IPython.display import clear_output
import statsmodels.api as sm # Linear regression
import shutil
import re


# create folder to results
def make_results_folders():
    path = "../../results"
    # If file in all_files, create check_folder to move files
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+"/alpha_a")
        os.makedirs(path+"/alpha_g")
        os.makedirs(path+"/N")
        os.makedirs(path+"/distributions")
        os.makedirs(path+"/network")
    else:
        pass

def all_properties_dataframe(N, dim, alpha_a, alpha_g):
    # Directory with all samples
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop"
    # All files with prop
    all_files = glob.glob(os.path.join(path,"*.csv"))

    # dataframe with all samples
    new_file = "/properties_set.txt"
    
    print(f"N={N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
    
    # if file exist, check if is necessery d update
    if(os.path.isfile(path + new_file) == True):
        df_name = pd.read_csv(path+"/filenames.txt", sep=' ')
        
        # Data to create DataFrame
        df_all = {"#short_path":[],
                "#diamater":[],
                "#ass_coeff":[]
                }
        file_names = []
        
        count = 0
        
        for file in all_files:
            cond = os.path.basename(file) in df_name["filename"].values
            if(cond == True):
                pass
            else:
                df = pd.read_csv(file, sep=',')

                df_all["#short_path"].append(df["#mean shortest path"].values[0])
                df_all["#diamater"].append(df["# diamater"].values[0])
                df_all["#ass_coeff"].append(df["#assortativity coefficient"].values[0])

                file_names.append(os.path.basename(file))
                count += 1

        if(count != 0):
            pall_df = pd.read_csv(path + "/properties_set.txt", sep=' ')
            pall_df = pd.concat([pall_df, pd.DataFrame(data = df_all)], ignore_index=True)
            df_name = pd.concat([df_name, pd.DataFrame(data = {"filename":file_names})], ignore_index=True)

            pall_df.to_csv(path+"/properties_set.txt", sep = ' ', index = False, mode = "w+")
            df_name.to_csv(path+"/filenames.txt", sep = ' ', index = False, mode = "w+")
                
        clear_output()  # Set wait=True if you want to clear the output without scrolling the notebook
    
    # file don't exist, create it
    elif(os.path.isfile(path + new_file) == False):
        n_values = 1
        
        # Data to create DataFrame
        df_all = {"#short_path":[],
                "#diamater":[],
                "#ass_coeff":[]
                }
        file_names = []

        for file in all_files:
            df = pd.read_csv(file, sep=',')
            df_all["#short_path"].append(df["#mean shortest path"].values[0])
            df_all["#diamater"].append(df["# diamater"].values[0])
            df_all["#ass_coeff"].append(df["#assortativity coefficient"].values[0])
            file_names.append(os.path.basename(file))
            print(f"{len(all_files)} files,{len(all_files) - n_values} remaning")
            n_values += 1
        df_prop = pd.DataFrame(data=df_all)
        df_name = pd.DataFrame(data={"filename":file_names})
        df_prop.to_csv(path+new_file,sep=' ', index=False, mode="w+")
        df_name.to_csv(path+"/filenames.txt",sep=' ', index=False, mode="w+")
        clear_output()  # Set wait=True if you want to clear the output without scrolling the notebook



# Open the folder string name and return (alpha_a_value, alpha_g_value)
def extract_alpha_values(folder_name):
    pattern = r"alpha_a_(-?\d+\.\d+)_alpha_g_(-?\d+\.\d+)"
    match = re.match(pattern, folder_name)
    if match:
        alpha_a = float(match.group(1))
        alpha_g = float(match.group(2))
        return (alpha_a, alpha_g)
    else:
        return None, None

# List all pair of (alpha_a,alpha_g) folders in (N,dim) folder
def list_all_folders(N,dim):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    set_parms = []

    for i in range(len(lst_folders)):
        set_parms.append(extract_alpha_values(lst_folders[i]))

    return set_parms

def list_all_folders_for_alpha_fixed(N,dim, alpha_a, alpha_g, alpha_g_variable):
    directory = f"../../data/N_{N}/dim_{dim}/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    lst_folders = lst_folders[0]
    
    set_parms = []

    for i in range(len(lst_folders)):
        term_ = len(lst_folders[i])
        if(term_ == 23):
            set_parms.append((lst_folders[i][8:11],lst_folders[i][20:]))
        if(term_== 24):
            set_parms.append((lst_folders[i][8:12],lst_folders[i][21:]))

    if(alpha_g_variable==True):
        alpha_g_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][0]==str(alpha_a)):
                alpha_g_V.append(set_parms[i][1])
        return alpha_g_V
    else:
        alpha_a_V = []
        for i in range(len(set_parms)):
            if(set_parms[i][1]==str(alpha_g)):
                alpha_a_V.append(set_parms[i][0])  
        return alpha_a_V

# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def create_all_properties_file(N,dim,alpha_a,alpha_g,alpha_g_variable):
    if(alpha_g_variable==True):
        mean_values = {"#alpha_g":[],"#short_mean":[],"#diamater_mean":[],
                       "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                       "#ass_coeff_err":[],"#n_samples":[]}
        
        for i in alpha_g:
            if(i==str(0.0)):
                pass
            else:
                df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{i}/prop/properties_set.txt", sep=' ')
                mean_values["#alpha_g"].append(i)
                
                mean_values["#short_mean"].append(df["#short_path"].mean())
                mean_values["#diamater_mean"].append(df["#diamater"].mean())
                mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
                
                mean_values["#short_err"].append(df["#short_path"].sem())
                mean_values["#diamater_err"].append(df["#diamater"].sem())
                mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

                mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_g', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(f"../../data/N_{N}/dim_{dim}/properties_all_alpha_a_{alpha_a}.txt",sep = ' ',index=False,mode="w+")
        
    else:       
        mean_values = {"#alpha_a":[],"#short_mean":[],"#diamater_mean":[],
                "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                "#ass_coeff_err":[],"#n_samples":[]}

        for i in alpha_a:
            df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{i}_alpha_g_{alpha_g}/prop/properties_set.txt", sep=' ')
            mean_values["#alpha_a"].append(i)
            
            mean_values["#short_mean"].append(df["#short_path"].mean())
            mean_values["#diamater_mean"].append(df["#diamater"].mean())
            mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
            
            mean_values["#short_err"].append(df["#short_path"].sem())
            mean_values["#diamater_err"].append(df["#diamater"].sem())
            mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

            mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_a', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(f"../../data/N_{N}/dim_{dim}/properties_all_alpha_g_{alpha_g}.txt",sep = ' ',index=False,mode="w+")


# Linear regression with errors in parameters
def linear_regression(X,Y,Erro_Y,Parameter):
    # Dados de exemplo
    x = X
    y = Y

    # Erros associados às medições no eixo y
    y_errors = Erro_Y

    # Calcular a regressão linear ponderada
    coefficients, cov_matrix = np.polyfit(x, y, deg=1, w=1/y_errors, cov=True)

    # Extrair os coeficientes e as incertezas
    slope = coefficients[0]
    intercept = coefficients[1]
    slope_error = np.sqrt(cov_matrix[0, 0])
    intercept_error = np.sqrt(cov_matrix[1, 1])
    
    # Return a, b, a_err, b_err
    if( Parameter == True):
        return slope, intercept, slope_error, intercept_error
    
    # Return y, where y = a*x + b
    else:
        return intercept + slope*x

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

def list_N_folders():
    directory = f"../../data/"
    lst_folders = []
    for root, dirs, files in os.walk(directory):
        lst_folders.append(dirs)
    dif = lst_folders[0]
    diff_N = [i[2:] for i in dif]
    run_second_time = 'operties_N' in diff_N
    
    if(run_second_time == True):
        diff_N.remove('operties_N')
    
    diff_N.remove(str(100000))
    diff_N.sort(key=int)
    #diff_N.remove(str(100000))
    return diff_N

# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def properties_dim_N(N, dim, alpha_g_variable):
    diff_head_N = [f"N_{i}" for i in N]    
    diff_head_err = [f"N_{i}_err" for i in N]
    
    dir_properties = f"../../data/properties_N/"
    conditional_ = os.path.exists(dir_properties)
    
    if(conditional_ == False):
        os.mkdir(dir_properties)
        os.mkdir(dir_properties+"/shortest_path/")
        os.mkdir(dir_properties+"/assortativity/")
        os.mkdir(dir_properties+"/diamater/")
    else:
        pass

    if(alpha_g_variable==True):
        diff_head_N = ["#alpha_g"] + diff_head_N
        header = diff_head_N + diff_head_err
        
        df_all_short = pd.DataFrame(columns=header)
        df_all_diamater = pd.DataFrame(columns=header)
        df_all_ass = pd.DataFrame(columns=header)
        
        alpha_a = 1.0
        j = 0
        for i in N:
            if(i==str(0.0)):
                pass
            else:
                file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_a_{alpha_a}.txt"
                df = pd.read_csv(file, sep=' ')
                
                df_all_short["#alpha_g"] = df["#alpha_g"]
                df_all_short[f"N_{i}"] = df["#short_mean"]
                df_all_short[f"N_{i}_err"] = df["#short_err"]
                
                df_all_diamater["#alpha_g"] = df["#alpha_g"]
                df_all_diamater[f"N_{i}"] = df["#diamater_mean"]
                df_all_diamater[f"N_{i}_err"] = df["#diamater_err"]

                df_all_ass["#alpha_g"] = df["#alpha_g"]
                df_all_ass[f"N_{i}"] = df["#ass_coeff_mean"]
                df_all_ass[f"N_{i}_err"] = df["#ass_coeff_err"]
            j += 1
        df_all_short.to_csv(f"../../data/properties_N/shortest_path/short_dim_{dim}_alpha_a_{alpha_a}.txt",sep = ' ',index=False,mode="w")
        df_all_diamater.to_csv(f"../../data/properties_N/diamater/diamater_dim_{dim}_alpha_a_{alpha_a}.txt",sep = ' ',index=False,mode="w")
        df_all_ass.to_csv(f"../../data/properties_N/assortativity/ass_dim_{dim}_alpha_a_{alpha_a}.txt",sep = ' ',index=False,mode="w")
    
    else:       
        diff_head_N = ["#alpha_a"] + diff_head_N
        header = diff_head_N + diff_head_err

        df_all_short = pd.DataFrame(columns=header)
        df_all_diamater = pd.DataFrame(columns=header)
        df_all_ass = pd.DataFrame(columns=header)

        alpha_g = 2.0
        j = 0

        for i in N:
            file = f"../../data/N_{i}/dim_{dim}/properties_all_alpha_g_{alpha_g}.txt"
            df = pd.read_csv(file, sep=' ')
            
            df_all_short["#alpha_a"] = df["#alpha_a"]
            df_all_short[f"N_{i}"] = df["#short_mean"]
            df_all_short[f"N_{i}_err"] = df["#short_err"]
            
            df_all_diamater["#alpha_a"] = df["#alpha_a"]
            df_all_diamater[f"N_{i}"] = df["#diamater_mean"]
            df_all_diamater[f"N_{i}_err"] = df["#diamater_err"]

            df_all_ass["#alpha_a"] = df["#alpha_a"]
            df_all_ass[f"N_{i}"] = df["#ass_coeff_mean"]
            df_all_ass[f"N_{i}_err"] = df["#ass_coeff_err"]

            j += 1
        df_all_short.to_csv(f"../../data/properties_N/shortest_path/short_dim_{dim}_alpha_g_{alpha_g}.txt",sep = ' ',index=False,mode="w")
        df_all_diamater.to_csv(f"../../data/properties_N/diamater/diamater_dim_{dim}_alpha_g_{alpha_g}.txt",sep = ' ',index=False,mode="w")
        df_all_ass.to_csv(f"../../data/properties_N/assortativity/ass_dim_{dim}_alpha_g_{alpha_g}.txt",sep = ' ',index=False,mode="w")


def all_properties_N(dim):
  diamater = pd.read_csv(f"../../data/properties_N/diamater/diamater_dim_{dim}_alpha_g_2.0.txt",sep=" ")
  shortest_path = pd.read_csv(f"../../data/properties_N/shortest_path/short_dim_{dim}_alpha_g_2.0.txt",sep=" ")
  assortativity = pd.read_csv(f"../../data/properties_N/assortativity/ass_dim_{dim}_alpha_g_2.0.txt",sep=" ")

  N_alpha_a = len(diamater) # Number of different alpha_a

  my_dictionary = {
    "alpha_a": [],
    "alpha_short": [],
    "alpha_short_err": [],
    "alpha_diameter": [],
    "alpha_diameter_err": [],
    "alpha_assortativity": [],
    "alpha_assortativity_err": []
  }
  X = np.array([np.log(int(i[2:])) for i in shortest_path.columns.tolist()[1:8]]) # N_values

  for i in range(N_alpha_a):
      Y_dia = diamater.iloc[i][1:8].values
      Y_ass = assortativity.iloc[i][1:8].values
      Y_short = shortest_path.iloc[i][1:8].values
      
      a_dia, err_a_dia = linear_regression(X, Y_dia, diamater.iloc[i][8:].values, Parameter=True)  
      a_ass, err_a_ass = linear_regression(X, Y_ass, assortativity.iloc[i][8:].values, Parameter=True)  
      a_short, err_a_short = linear_regression(X, Y_short, shortest_path.iloc[i][8:].values, Parameter=True)  
      
      my_dictionary["alpha_a"].append(shortest_path.iloc[i][0])
      my_dictionary["alpha_short"].append(a_short)
      my_dictionary["alpha_short_err"].append(err_a_short)
      my_dictionary["alpha_diameter"].append(a_dia)
      my_dictionary["alpha_diameter_err"].append(err_a_dia)
      my_dictionary["alpha_assortativity"].append(a_ass)
      my_dictionary["alpha_assortativity_err"].append(err_a_ass)

  coeff_properties = pd.DataFrame(data=my_dictionary)
  sorted_df = coeff_properties.sort_values(by='alpha_a', key=lambda col: col.astype(float))  # Sort by converting to float
  sorted_df.to_csv(f"../../data/properties_dim_{dim}.csv",index=False,mode="w")

def filter_N_properties(alpha_filter,properties):
    # All index where alpha_a in all_alpha_a dataframe
    all_alpha_a = [properties.iloc[i,0] in alpha_filter for i in range(len(properties))]
    # Select index with alpha_a values
    index = [i for i in range(len(all_alpha_a)) if all_alpha_a[i]==True]
    # Values of properties
    N_values = [int(i[2:]) for i in properties.columns.values.tolist()[1:8]]

    properties_values = []
    err_properties_path = []
    for j in range(len(index)):
        properties_values.append(properties.iloc[index[j]][1:8].values)
        err_properties_path.append(properties.iloc[index[j]][8:].values)
    return N_values, properties_values, err_properties_path

def filter_N_linear_regression(alpha_filter,properties):
    # All index where alpha_a in all_alpha_a dataframe
    all_alpha_a = [properties.iloc[i,0] in alpha_filter for i in range(len(properties))]
    # Select index with alpha_a values
    index = [i for i in range(len(all_alpha_a)) if all_alpha_a[i]==True]
    # Values of properties
    N_values = [int(i[2:]) for i in properties.columns.values.tolist()[1:8]]

    properties_values = []
    err_properties_path = []
    for j in range(len(index)):
        properties_values.append(properties.iloc[index[j]][1:8].values)
        err_properties_path.append(properties.iloc[index[j]][8:].values)
    return N_values, properties_values, err_properties_path


# alpha_filter: the values of alpha_a that you wanna
# properties: dataframe with all properties
# header: properties that you wanna studie (#short_mean, #diamater_mean,#ass_coef)
# err_header: error in properties (#short_err, #diamater_err, #ass_coeff_err)

def filter_alpha_a_properties(alpha_filter,properties,header,err_header):
    # All index where alpha_a in all_alpha_a dataframe
    all_alpha_a = [properties.iloc[i,0] in alpha_filter for i in range(len(properties))]
    # Select index with alpha_a values
    index = [i for i in range(len(all_alpha_a)) if all_alpha_a[i]==True]
    # Values of properties
    
    properties_values = []
    err_properties_path = []
    for j in range(len(index)):
        properties_values.append(properties.iloc[index[j]][header])
        err_properties_path.append(properties.iloc[index[j]][err_header])
    return properties_values, err_properties_path

def reset_files(N,dim,alpha_g,alpha_a):
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/properties_set.txt"
    if(os.path.exists(path)==True):
        os.remove(path)
    else:
        pass

def filter_alpha_g(alpha_g, propertie, err_propertie):
    alpha_g_filter, propertie_filter,propertie_err_filter = [],[],[]
    
    for i in range(len(alpha_g)):
        if(alpha_g[i]>1.0):
            alpha_g_filter.append(alpha_g[i])
            propertie_filter.append(propertie[i])
            propertie_err_filter.append(err_propertie[i])
    alpha_g_filter, propertie_filter, propertie_err_filter = np.array(alpha_g_filter), np.array(propertie_filter), np.array(propertie_err_filter)
    return alpha_g_filter, propertie_filter, propertie_err_filter

# reset folders properties
def reset_folders(N, dim, alpha_a, alpha_g):
    path_check = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop_check"
    path_prop = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/"
    
    all_files_check = glob.glob(os.path.join(path_check,"*.csv"))
    
    if(len(all_files_check)!=0):
        
        for file in all_files_check:
            shutil.move(file, path_prop + os.path.basename(file))
    else:
        pass

# Create datarframe with alpha_g fixed and alpha_a variable or the other way around (N fixed)
def create_all_surface(N,dim,alpha_a,alpha_g,alpha_g_variable):

    path = f"../../data/surface/N_{N}/dim_{dim}"  
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    if(alpha_g_variable==True):
        mean_values = {"#alpha_g":[],"#short_mean":[],"#diamater_mean":[],
                       "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                       "#ass_coeff_err":[],"#n_samples":[]}
        for i in alpha_g:
            if(i==str(0.0)):
                pass
            else:
                df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}alpha_g{i}/prop/properties_set.txt", sep=' ')
                mean_values["#alpha_g"].append(i)
                
                mean_values["#short_mean"].append(df["#short_path"].mean())
                mean_values["#diamater_mean"].append(df["#diamater"].mean())
                mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
                
                mean_values["#short_err"].append(df["#short_path"].sem())
                mean_values["#diamater_err"].append(df["#diamater"].sem())
                mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

                mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_g', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_a_{alpha_a}.txt", sep = ' ', index = False, mode="w+")
        
    else:       
        mean_values = {"#alpha_a":[],"#short_mean":[],"#diamater_mean":[],
                "#ass_coeff_mean":[],"#short_err":[],"#diamater_err":[],
                "#ass_coeff_err":[],"#n_samples":[]}

        for i in alpha_a:
            df = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{i}alpha_g{alpha_g}/prop/properties_set.txt", sep=' ')
            mean_values["#alpha_a"].append(i)
            
            mean_values["#short_mean"].append(df["#short_path"].mean())
            mean_values["#diamater_mean"].append(df["#diamater"].mean())
            mean_values["#ass_coeff_mean"].append(df["#ass_coeff"].mean())
            
            mean_values["#short_err"].append(df["#short_path"].sem())
            mean_values["#diamater_err"].append(df["#diamater"].sem())
            mean_values["#ass_coeff_err"].append(df["#ass_coeff"].sem())

            mean_values["#n_samples"].append(len(df["#diamater"]))
        
        df_all = pd.DataFrame(data=mean_values)
        sorted_df = df_all.sort_values(by='#alpha_a', key=lambda col: col.astype(float))  # Sort by converting to float
        sorted_df.to_csv(path + f"/all_alpha_g_{alpha_g}.txt", sep = ' ', index=False, mode="w+")