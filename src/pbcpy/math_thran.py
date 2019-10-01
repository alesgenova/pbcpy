import numpy as np

#pythran export multiply(float[:] or float[,] or float[,,] or float[,,,], 
# float[:] or float[,] or float[,,] or float[,,,]) 
#pythran export add(float[:] or float[,] or float[,,] or float[,,,], 
# float[:] or float[,] or float[,,] or float[,,,]) 
#pythran export power(float[:] or float[,] or float[,,] or float[,,,], 
# float or float[:] or float[,] or float[,,] or float[,,,]) 
def multiply(arr1,arr2):
    results = np.multiply(arr1, arr2)
    return results

def add(arr1,arr2):
    results = np.add(arr1, arr2)
    return results

def power(arr1,x):
    results = np.power(arr1, x)
    return results
