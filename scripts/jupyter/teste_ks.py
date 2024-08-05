import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.stats import kstest
import pandas as pd
import math

def find_order_of_magnitude(number):
    order = int(math.floor(math.log10(abs(number))))
    return abs(order)

# Generate data from a q-exponential distribution
q_parameter = 1.140
b_parameter = 1.01

N = 10**5
dim = 4
alpha_a = 5.0
alpha_g = 2.0
#full_dataset = np.random.exponential(scale=b_parameter, size=size) ** (1 / (1 - q_parameter))
dataset = pd.read_csv(f"../../data/N_{N}/dim_{dim}/alpha_a_{alpha_a}_alpha_g_{alpha_g}/all_files/distri_all.csv")

x_values = dataset["k"].values
full_dataset = dataset["pk"].values

k = []
pk = []

order = 3

for i in range(len(x_values)):
    if(find_order_of_magnitude(full_dataset[i])<=order):
        k.append(x_values[i])
        pk.append(full_dataset[i])

size = len(pk)

# Define the q-exponential distribution function
def normalized_constant(x, q, b):
    lst_aux = []
    for i in range(len(x)):
        lst_aux.append((1 - ( 1 - q) * ( x[i] / b ))**( 1  / ( 1 - q )))        
    
    return 1/sum(lst_aux)

def q_exp(x, q, b):
    A = normalized_constant(x, q, b)
    norm_prob = []
    for i in range(len(x)):
        term = A * (1 - (1 - q) * ( x[i] / b )) ** (1 / (1 - q))
        norm_prob.append(term)
    return norm_prob

def q_exp_c(p_x):
    p_c = np.zeros(len(p_x))
    p_c[0] = sum(p_x)

    for i in range(1,len(p_x)):
        p_c[i] = p_c[i-1] - p_x[i]
    return p_c

# Create a model using the custom q-exponential function
model = Model(q_exp)

# Set initial parameter values
params = model.make_params(q=q_parameter, b=b_parameter)

# Fit the model to the data
result = model.fit(pk, params, x=k)

# Get the fitted parameters
fitted_q = result.params['q'].value
fitted_b = result.params['b'].value

A = normalized_constant(x_values, fitted_q, fitted_b)
print(A)

#pk_real = q_exp(A, k, fitted_q, fitted_b)
pk_real = q_exp(k, fitted_q, fitted_b)
#q_exp_real = q_exp_c(pk_real)

# Perform KS test on the fitted q-exponential distribution
#ks_statistic, ks_p_value = kstest(pk, lambda k: q_exp_c(k, fitted_q, fitted_b))
ks_statistic, ks_p_value = kstest(pk, q_exp(k,fitted_q,fitted_b))

# Display the results
print(result.fit_report())
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {ks_p_value}")
print(max(k))
#Plot the original dataset and the fitted q-exponential distribution
#x_values = np.linspace(min(full_dataset), max(full_dataset), 100)
plt.plot()
plt.plot(k, q_exp(k, fitted_q, fitted_b), 'b--', label='Fitted q-Exponential')
plt.plot(k, pk, 'ro', label='real data')
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
