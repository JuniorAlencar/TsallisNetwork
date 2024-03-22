from src.AssortativityFunctions import *

N = 10**5
dim = [1,2,3,4]
alpha_a_v = [0.0, 6.0]
#alpha_g_v = [1.0, 2.0, 3.0, 4.0, 5.0]
#alpha_g_v = [4.0, 5.0]

#alpha_a_f = [1.0]
alpha_g_f = [2.0]

#assortativity_multithread(dim, alpha_a_f, alpha_g_v)
assortativity_multithread(dim, alpha_a_v, alpha_g_f)

permission_run(alpha_a_v, alpha_g_f)
#permission_run(alpha_a_f, alpha_g_v)
