from src.PropertiesFunctions import *
# --------------------------------------------------------------------->
# FunctionsFile defined in folder ./src

#mean_values(N, dim, alpha_a, alpha_g, error)
#-> generate mean properties:
# error: boolean variable, if True return error properties
# Error False, return -->
# shortest_path_mean, diamater_mean, assortativity_mean
# Error True, return -->
# shortest_path_mean, diamater_mean, assortativity_mean
# error_shortest_path, error_diamater, error_assortativity
# --------------------------------------------------------------------->
# Gamma(alpha_a,d)
# Return gamma values
# --------------------------------------------------------------------->
# Lambda(alpha_a, d)
# Return lambda values
# --------------------------------------------------------------------->
#all_data_mean(N, alpha_a, alpha_g, Alpha_A)
# -> Generate matrix with mean of all samples
# Alpha_A boolean variable
# if True  return properties(alpha_a, d)
# if False return properties(alpha_g, d)

# Alpha_A True:
# Return -> 
# properties.shape(dim, alpha_a)
# shortest(alpha_a, dim), diamater(alpha_a, dim), coefAss(alpha_a, dim)
# alpha_a, dim

# Alpha_A False:
# Return -> 
# properties.shape(dim, alpha_g)
# shortest(alpha_g, dim), diamater(alpha_g, dim), coefAss(alpha_g, dim)
# alpha_g, dim
# --------------------------------------------------------------------->
# Loading all values of alpha_a and alpha_g (reading directories)
N = [5000,10000,20000,40000,80000,160000,320000]
dim = [1,2,3,4]
alpha_g = 2.0
alpha_a = list_all_folders_for_alpha_fixed(320000,dim[0],alpha_g_variable=False)

'''
for i in N:
	for j in dim:
		for k in alpha_a:
			all_properties_dataframe(i,j,k,alpha_g)
'''

for i in N:
	for j in dim:
		create_all_properties_file(i,j,alpha_a,alpha_g,alpha_g_variable=False)

#color = ["black","#03AC13","#00019a","#fe0000"]
