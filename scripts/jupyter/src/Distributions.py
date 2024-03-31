import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import glob
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from IPython.display import clear_output
import gzip
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sklearn.model_selection import train_test_split
import shutil
import math


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

# Return tuple (alpha_a,alpha_g)
def list_all_folders(N,dim):
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
        if(term_ == 24):
            set_parms.append((lst_folders[i][8:12],lst_folders[i][21:]))
    return set_parms

def distribution(N, dim, alpha_a, alpha_g, degree, save):
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
    
    if(save == True):
        # Save distribution (or update)
        distri_dataframe = pd.DataFrame(data={"k":k_real,"pk":p_real})
        distri_dataframe['pk'] = distri_dataframe['pk'].apply(lambda x: format(x, '.2e'))
        distri_dataframe.to_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/distri_linear_all.csv",mode="w",index=False)
        return k_real,p_real
    else:
        return k_real,p_real
 
    

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(N, dim, alpha_a, alpha_g, counter_dict, bin_count, save):

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
    
    if(save==True):
        distri_dataframe = pd.DataFrame(data={"k":k,"pk":Pk})
        distri_dataframe['pk'] = distri_dataframe['pk'].apply(lambda x: format(x, '.2e'))
        distri_dataframe.to_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/distri_log_all.csv",mode="w",index=False)
        return k,Pk
    else:
        return k,Pk

    
def ln_q(k, pk, q, eta):
    k_values = np.zeros(len(k))
    for i in range(len(k)):
        k_values[i] = (1-(1-q)*(k[i]/eta))**(1/(1-q))
    P0 = 1/sum(k_values)
    return ((pk/P0)**(1-q)-1)/(1-q)

def reader_degree_all(N, dim, alpha_a, alpha_g):
    path_degree = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/degree_all.txt"
    df = pd.read_csv(path_degree)
    return df["degree"].values

def deletar_pastas(caminho_pasta):
    # Verifica se o caminho existe
    if os.path.exists(caminho_pasta):
        # Percorre todas as pastas e arquivos no caminho especificado
        for item in os.listdir(caminho_pasta):
            item_path = os.path.join(caminho_pasta, item)
            
            # Se for um diretório, chama a função recursivamente
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            # Se for um arquivo, exclui o arquivo
            else:
                os.remove(item_path)
        
        # Após excluir todos os itens, remove a pasta principal
        shutil.rmtree(caminho_pasta)
        print(f'A pasta {caminho_pasta} e seu conteúdo foram excluídos com sucesso.')
    else:
        print(f'O caminho {caminho_pasta} não existe.')


def q(alpha_a,d):
    if(0 <= alpha_a/d <= 1):
        return 4/3
    else:
        return round((1/3)*np.exp(1-alpha_a/d)+1.0,4)

def kappa(alpha_a,d):
    if(0 <= alpha_a/d <= 1):
        return 0.3
    else:
        return round(-1.15*np.exp(1-alpha_a/d)+1.45,4)

def reset_gml_folder(N,dim,alpha_a,alpha_g):
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/check"
    path_o = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/gml/"
    
    all_files = glob.glob(os.path.join(path, "*.gml.gz"))
    
    for file in all_files:
        shutil.move(file, path_o + os.path.basename(file))

def find_order_of_magnitude(number):
    order = int(math.floor(math.log10(abs(number))))
    return abs(order)    

def all_degree_GML(N, dim, alpha_a, alpha_g):
    path = f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}"
    all_files = glob.glob(os.path.join(path + "/gml","*.gml.gz"))

    if not os.path.exists(path + "/all_files"):
        os.makedirs(path + "/all_files")
    else:
        pass

    if(os.path.exists(path + "/all_files/degree_all.txt") == True):
        print(f"N={N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
        file_data = pd.read_csv(path+"/all_files/file_name_all.txt", sep=" ")
        # Create new dictionary aux to update (if necessary)
        file_names = {"files": [] }
        degrees = []
        
        count = 0
        degrees_dict = defaultdict(list)
        
        for file in all_files:
            file_name = os.path.basename(file)
        
            # Check if file was used
            cond = file_name in file_data["files"].values    
            
            # If the file is dataframe, pass
            if(cond == True):
                pass
            # Else update
            elif(cond == False):
                print("not update")
                
                file_names["files"].append(os.path.basename(file))
                degrees_dict = defaultdict(list)
                

                with open(file, 'rb') as file_in:
                    # decompress gzip
                    with gzip.GzipFile(fileobj=file_in, mode='rb') as gzip_file:
                        for line in gzip_file:
                            # decode file
                            line = line.decode('utf-8')
                            if(line[:6]=="degree"):
                                degrees.append(int(line[7:]))
                count += 1        
        
            degrees_dict['degree'].extend(degrees)
        
        if(count != 0):
            degree_data = pd.read_csv("degree.txt", sep=" ")
            
            degree_data = pd.concat([degree_data, pd.DataFrame(degrees_dict)], ignore_index=True)
            file_data = pd.concat([file_data, pd.DataFrame(data=file_names)], ignore_index=True)
            
            degree_data.to_csv(path + "/all_files/degree_all.txt", sep = ' ', index=False, mode="w+")
            file_data.to_csv(path + "/all_files/file_name_all.txt", sep = ' ',  index=False, mode="w+")        
        clear_output()  # Set wait=True if you want to clear the output without scrolling the notebook
        
    elif(os.path.exists(path + "/all_files/degree_all.txt") == False):
        print(f"N={N}, dim = {dim}, alpha_a = {alpha_a}, alpha_g = {alpha_g}")
        
        file_names = { "files": [] }
        degrees = []
        num_files = 1
        
        for file in all_files: 
            file_names["files"].append(os.path.basename(file))
            
            with open(file, 'rb') as file_in:
                # decompress gzip
                with gzip.GzipFile(fileobj=file_in, mode='rb') as gzip_file:
                    for line in gzip_file:
                        # decode file
                        line = line.decode('utf-8')
                        if(line[:6]=="degree"):
                            degrees.append(int(line[7:]))
            print(f"N_files = {len(all_files)}, {len(all_files)-num_files} remaning")
            num_files += 1
            
        file_df = pd.DataFrame(data=file_names)
        degree_df = pd.DataFrame(data={"degree":degrees})
        
        degree_df.to_csv(path + "/all_files/degree_all.txt", sep = ' ', index=False,mode='w+')
        file_df.to_csv(path + "/all_files/file_name_all.txt", sep = ' ', index=False,mode='w+')
        clear_output()  # Set wait=True if you want to clear the output without scrolling the notebook