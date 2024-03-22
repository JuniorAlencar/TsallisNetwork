import numpy as np
import glob
import os

def ScriptGenerate(N,alpha_a,alpha_g,dim,NumSamples):
    filename = f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}.sh"

    a = "#!/bin/bash\n"
    b = f"#SBATCH -J a{alpha_a:.1f}_g{alpha_g:.1f}_d{dim}\n"
    c = "#SBATCH --mail-user=junioralencar@fisica.ufc.br\n"
    d = "#SBATCH --ntasks=1\n"
    e = "#SBATCH --mail-type=ALL\n"

    g = "x=0\n"
    h = f"n_samples={NumSamples}\n"
    i = "while [ $x -le $n_samples ]\n"
    j = "do\n\t"
    m = f"srun ../build/exe1 ../parms/N_{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}.json\n\t"
    q = "x=$(( $x + 1 ))\n"
    v = "done"

    newpath = "../multithread/"
    
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    list_for_loop = [a,b,c,d,e,g,h,i,j,m,q,v]
    l = open(newpath + filename, "w") # argument w: write if don't exist file

    for k in list_for_loop:
        l.write(k)
    l.close()

def JsonGenerate(N,alpha_a,alpha_g,dim):
    filename = f"N_{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}.json"
    a = "{\n"
    b = "\"comment\": \"use seed= -1 for random seed\",\n"
    c = f"\"num_vertices\": {N},\n"
    d = f"\"alpha_a\": {alpha_a:.6f},\n"
    e = f"\"alpha_g\": {alpha_g:.6f},\n"
    f = "\"r_min\": 1,\n"
    g = "\"r_max\": 10000000,\n"
    h = f"\"dim\": {dim},\n"
    i = "\"seed\": -1\n"
    j = "}"
    
    newpath = "../../parms/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    list_for_loop = [a,b,c,d,e,f,g,h,i,j]
    x = open(newpath + filename, "w")

    for k in list_for_loop:
        x.write(k)
    x.close()

def text_terminal():
    newpath = "../../data/"
    
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    os.chdir(newpath)
    
    filename = "../scripts/run_all_script.txt"
    
    path = "../scripts/multithread"
    all_files = glob.glob(os.path.join(path,"*.sh"))

    l = open(filename, "w") # argument w: write if don't exist file
    a = 0
    for file in all_files:
        a += 1
        if(a==len(all_files)):
            file_ = f"sbatch\t {file}"
        else:
            file_ = f"sbatch\t {file} \t && \t"
        l.write(file_)

    l.close()
