import numpy as np
import glob
import os

def JsonGenerate(N, alpha_a,alpha_g,dim):
    filename = f"N{N}_a{alpha_a:.2f}_g{alpha_g:.2f}_d{dim}.json"
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
    
    newpath = f"../../parms_pc_{N}/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    list_for_loop = [a,b,c,d,e,f,g,h,i,j]
    x = open(newpath + filename, "w")

    for k in list_for_loop:
        x.write(k)
    x.close()

def multithread_pc(N,NumSamples):
    filename = f"N_{N}_multithread_pc.sh"

    a = "#!/bin/bash\n\n"
    
    b = "# Define uma função que contêm o código para rodar em paralelo\n"
    
    c = "run_code() {\n\t"
    d = f"time ../build/exe1 ../parms_pc_{N}/$1\n"
    e = "}\n"
    f = "# Exportar a função usando o módulo Parallel\n"
    g = "export -f run_code\n\n"
    
    path_d = f"../../parms_pc_{N}"
    all_files = glob.glob(os.path.join(path_d,"*.json"))
    list_of_arguments = [V[2] for V in os.walk(path_d)][0]
    list_of_arguments = str(list_of_arguments)
    list_of_arguments = list_of_arguments.replace(',', '')
    
    h = f"arguments=(" 
    i = list_of_arguments[1:-1] + ")\n"
    j = "x=0\n"
    k = f"n_samples={NumSamples}\n"
    l = f"while [ $x -le $n_samples ]\n"
    m = "do\n\t"
    n = "parallel run_code :::\t" +  """ "${arguments[@]}"  """ "\n\t"
    o = "x=$(( $x + 1))\n"
    p = "done"

    
    list_for_loop = [a,b,c,d,e,g,h,i,j,k,l,m,n,o,p]
    l = open("../" + filename, "w") # argument w: write if don't exist file

    for k in list_for_loop:
        l.write(k)
    l.close()

def permission_run(N):
	os.system(f"chmod 700 ../N_{N}_multithread_pc.sh")
