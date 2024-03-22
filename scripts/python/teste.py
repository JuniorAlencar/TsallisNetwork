import numpy as np
# Taking space-separated numbers as input and converting them into a list of integers
num_list = list(map(int, input("Enter numbers separated by space: ").split()))
result = np.array(num_list)*3
print("List of numbers:", result)
