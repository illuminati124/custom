import math
import scipy
import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.spatial.distance import correlation as cp


'''
# FUNCTIONS to be used:
    - cos_sim(vec_1[][], vec_2[][])
    - sqrt_cos_sim(vec_1[][], vec_2[][])
    - correlation_similarity(vec_1[][], vec_2[][])
'''

# Cosine Similarity
def find_cosine_similarity(vec_1, vec_2):
    return np.dot(vec_1, vec_2)/(norm(vec_1)*norm(vec_2))

def cos_sim(vec_1, vec_2):
    a = len(vec_1)
    b = len(vec_2)
    
    ans = []
    for i in range(0, a):
        ans.append([])
        for j in range(0, b):
            ans[i].append(find_cosine_similarity(vec_1[i], vec_2[j]))
    
    return np.array(ans)


# ISC (Improved Sqrt Cosine Similarity)
def get_sqrt(a, b):
    if a*b >= 0:
        return (a*b)**0.5
    else:
        return -(abs(a*b))**0.5
    
def find_sqrt_cosine_similarity(vec_1, vec_2):
    return (np.dot(np.where(vec_1 < 0, -np.sqrt(-vec_1), np.sqrt(vec_1)), np.where(vec_2 < 0, -np.sqrt(-vec_2), np.sqrt(vec_2)))) / (np.sqrt(np.sum(abs(vec_1))) * np.sqrt(np.sum(abs(vec_2))))
    
def sqrt_cos_sim(vec_1, vec_2):
    a = len(vec_1)
    b = len(vec_2)
    
    ans = []
    for i in range(0, a):
        ans.append([])
        for j in range(0, b):
            ans[i].append(find_sqrt_cosine_similarity(vec_1[i], vec_2[j]))
    
    return np.array(ans)


# Pearson Correlation Similarity
def correlation_similarity(vec_1, vec_2):
    a = len(vec_1)
    b = len(vec_2)
    
    ans = []
    for i in range(0, a):
        ans.append([])
        for j in range(0, b):
            ans[i].append(1 - cp(vec_1[i], vec_2[j]))
    
    return np.array(ans)