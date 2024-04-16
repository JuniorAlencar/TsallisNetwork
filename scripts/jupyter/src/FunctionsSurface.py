import pandas as pd
import glob
import os
import numpy as np
from IPython.display import clear_output
import math
import re
import shutil

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
        # Run code if /prop folder exist
        try: 
            for file in all_files:
                cond = os.path.basename(file) in df_name["filename"].values
                if(cond == True):
                    pass
                else:
                    # Read CSV file into a DataFrame
                    try:
                        df = pd.read_csv(file, sep=',')
                        df_all["#short_path"].append(df["#mean shortest path"].values[0])
                        df_all["#diamater"].append(df["# diamater"].values[0])
                        df_all["#ass_coeff"].append(df["#assortativity coefficient"].values[0])

                        file_names.append(os.path.basename(file))
                        count += 1
                    
                    except pd.errors.EmptyDataError:
                        # File is empty, remove it
                        os.remove(file)
        
        # Remove folder if /prop folder don't exist
        except OSError:
            # Folder empty, remove it
            shutil.rmtree(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}")
                    

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
        
        # Run code if /prop folder exist
        try:
            for file in all_files:
                
                try:
                    df = pd.read_csv(file, sep=',')
                    df_all["#short_path"].append(df["#mean shortest path"].values[0])
                    df_all["#diamater"].append(df["# diamater"].values[0])
                    df_all["#ass_coeff"].append(df["#assortativity coefficient"].values[0])
                    file_names.append(os.path.basename(file))
                    print(f"{len(all_files)} files,{len(all_files) - n_values} remaning")
                    n_values += 1

                except pd.errors.EmptyDataError:
                    os.remove(file)
            df_prop = pd.DataFrame(data=df_all)
            df_name = pd.DataFrame(data={"filename":file_names})
            df_prop.to_csv(path+new_file,sep=' ', index=False, mode="w+")
            df_name.to_csv(path+"/filenames.txt",sep=' ', index=False, mode="w+")
            clear_output()  # Set wait=True if you want to clear the output without scrolling the notebook       
                
        # Remove folder if /prop folder don't exist
        except OSError:
            # Folder empty, remove it
            shutil.rmtree(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}")

# Get the interval values
def filter_dataframe(dataframe):
    alpha_a_intervals = [round(i,2) for i in range(0,10)]
    #alpha_g_intervals = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    df = pd.DataFrame(columns=dataframe.columns.tolist())
    # alpha_a_intervals = get_intervals(dataframe)
    # alpha_g_intervals = get_intervals(dataframe, param='alpha_g')
    # Creates an empty array for the grid
    param_means_grid = []
    # Iterate over alpha_a and alpha_g and takes the mean of a given param

    for i in range(len(alpha_a_intervals)):
        param_row = []
        mask_alpha_a = dataframe[dataframe['alpha_a'] == alpha_a_intervals[i]]
        #df_combined = df.append(mask_alpha_a, ignore_index=True)
        df = pd.concat([df, mask_alpha_a], ignore_index=True)
    
    return df

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
            set_parms.append((lst_folders[i][8:10],lst_folders[i][20:]))
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

# Create and save dataframe with Beta values (Propertie = Beta*log10(N) + Gamma)
def beta_all(dataframe_filter, List_N):    
    # Dict to load beta_values with parameters
    properties_dict = {"alpha_a":[], "alpha_g":[], "dim":[],
                        "beta_short":[], "beta_diameter":[],"beta_assortativity":[],
                    "beta_short_err":[], "beta_diameter_err":[],"beta_assortativity_err":[]}

    # Create tuple with alpha_a, alpha_g, dim from dataframe_filter
    index_dict_data = [(i, j, k) for i,j,k in zip(dataframe_filter['alpha_a'],
                                                dataframe_filter['alpha_g'],
                                                dataframe_filter['dim'])]
    # Extract unique tuples using set
    unique_tuples = set(index_dict_data)

    # List of N_values
    


    # run for unique values of (alpha_a, alpha_g, dim)
    for element in unique_tuples:
        
        # Append parameter values in dictionary
        properties_dict["alpha_a"].append(element[0])
        properties_dict["alpha_g"].append(element[1])
        properties_dict["dim"].append(element[2])
        
        # lists to help store Betas and errors
        short_mean, short_mean_err = [], []
        diameter_mean, diameter_mean_err = [], []
        ass_coeff_mean, ass_coeff_mean_err = [], []
        
        # Run the code to each n in List_N
        for n in List_N:
            # Select the index with unique (alpha_a, alpha_g, dim) for each n
            idx_values_N = dataframe_filter[(dataframe_filter['alpha_a'] == element[0]) & 
                            (dataframe_filter['alpha_g'] == element[1]) &
                            (dataframe_filter['dim'] == element[2]) & 
                            (dataframe_filter['n_size'] == n)].index[0]
            
            # Append properties in lists
            short_mean.append(dataframe_filter["short_mean"].iloc[idx_values_N])
            short_mean_err.append(dataframe_filter["short_err"].iloc[idx_values_N])

            diameter_mean.append(dataframe_filter["diameter_mean"].iloc[idx_values_N])
            diameter_mean_err.append(dataframe_filter["diameter_err"].iloc[idx_values_N])

            ass_coeff_mean.append(dataframe_filter["ass_coeff_mean"].iloc[idx_values_N])
            ass_coeff_mean_err.append(dataframe_filter["ass_coeff_err"].iloc[idx_values_N])
        
        # Calcule of linear regression with errors (beta, beta_err)
        beta_short = linear_regression(np.log10(List_N), 
                                                    np.array(short_mean), np.array(short_mean_err), Parameter = True)
        
        beta_diameter = linear_regression(np.log10(List_N), 
                                                    np.array(diameter_mean), np.array(diameter_mean_err), Parameter = True)
        
        beta_ass = linear_regression(np.log10(List_N), 
                                                    np.array(ass_coeff_mean), np.array(ass_coeff_mean_err), Parameter = True)
        
        # Append values in dictionary
        properties_dict["beta_short"].append(beta_short[0])
        properties_dict["beta_short_err"].append(beta_short[2])

        properties_dict["beta_diameter"].append(beta_diameter[0])
        properties_dict["beta_diameter_err"].append(beta_diameter[2])

        properties_dict["beta_assortativity"].append(beta_ass[0])
        properties_dict["beta_assortativity_err"].append(beta_ass[2])

    # Create and save DataFrame with dictionary data
    df = pd.DataFrame(data=properties_dict)
    df.to_csv(f"../../data/all_beta.csv", sep = ',', mode = "w+", index = False)
    return df

# Create and save dataframe with N* values (-> R(N*) = 0 <-)
def assortativity_N(dataframe_filter, List_N):
    # Dict to load beta_values with parameters
    properties_dict = {"alpha_a":[], "alpha_g":[], "dim":[],
                        "N*":[]}

    # Create tuple with alpha_a, alpha_g, dim from dataframe_filter
    index_dict_data = [(i, j, k) for i,j,k in zip(dataframe_filter['alpha_a'],
                                                dataframe_filter['alpha_g'],
                                                dataframe_filter['dim'])]
    # Extract unique tuples using set
    unique_tuples = set(index_dict_data)

    # run for unique values of (alpha_a, alpha_g, dim)
    for element in unique_tuples:
        
        # Lists to store assortativity and errors
        ass_coeff_mean, ass_coeff_mean_err = [], []    
        
        # Append parameter values in dictionary
        properties_dict["alpha_a"].append(element[0])
        properties_dict["alpha_g"].append(element[1])
        properties_dict["dim"].append(element[2])
        
        for n in List_N:
            
            # Select the index with unique (alpha_a, alpha_g, dim) for each n
            idx_values_N = dataframe_filter[(dataframe_filter['alpha_a'] == element[0]) & 
                            (dataframe_filter['alpha_g'] == element[1]) &
                            (dataframe_filter['dim'] == element[2]) & 
                            (dataframe_filter['n_size'] == n)].index[0]

            ass_coeff_mean.append(dataframe_filter["ass_coeff_mean"].iloc[idx_values_N])
            ass_coeff_mean_err.append(dataframe_filter["ass_coeff_err"].iloc[idx_values_N])
        
        beta_ass = linear_regression(np.log10(List_N), 
                                                    np.array(ass_coeff_mean), np.array(ass_coeff_mean_err), Parameter = True)
        
        # N* = 10**(-Gamma/Beta), where: R = Beta*log10(N) + Gamma. -> R(N*) = 0 <-
        properties_dict["N*"].append(10**(-beta_ass[1]/beta_ass[0]))

    df = pd.DataFrame(data=properties_dict)
    df.to_csv(f"../../data/all_N.csv", sep = ',', mode = "w+", index = False)
    return df

# Return order in number, b (number = c*10**b)
def find_order_of_magnitude(number):
    order = int(math.floor(math.log10(abs(number))))
    return abs(order)  

# Function to transfer data to Cleber
def copy_files_cleber(N, dim, alpha_a, alpha_g):
    my_file = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop/properties_set.txt"
    new_path = f"../../../raw_N_{N}/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/prop"
    
    # Create folder with file
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        # Copy file to /raw_N_{N}
        shutil.copy(my_file, new_path)
    else:
        pass