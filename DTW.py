from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
import numpy as np

'''Compute distance matix given my query and my input string'''
def computeDistanceMatrix(x,y):
	r,c = len(x), len(y)
	distance_matrix = np.zeros(shape=(r,c))  
	for i in range(r):
		for j in range(c):
			if(x[i]!=y[j]):
				distance_matrix[i][j] = 1
	return distance_matrix

'''Compute the cost matrix and trace the paths from the cost matrix'''
def dtw(distance_matrix):
	#import pdb;pdb.set_trace()
	m, n = np.shape(distance_matrix)
	cost = np.zeros(shape = np.shape(distance_matrix))
	cost[0,:] = distance_matrix[0,:]
	cost[:,0] = np.cumsum(distance_matrix[:,0])
	#Traceback initialization
	
	traceback_indexes = np.zeros(shape=(m, n))
	#traceback_indexes[0, 1:] = inf
	#traceback_indexes[1:,0] = inf
	for i in range(1,m):
		for j in range(1,n):
			minimum_value = min([cost[i-1,j-1],cost[i-1,j],cost[i,j-1]])
			traceback_indexes[i,j] = np.argmin([cost[i-1,j-1],cost[i-1,j],cost[i,j-1]])
			cost[i,j] = distance_matrix[i,j] + minimum_value
	return cost,traceback_indexes

'''Traceback function without applying the minimum cost heuristic in the last row'''
# def traceback(cost_matrix, traceback_indexes):

# 	 #Traceback indices are 0 for diagnol, 1 for vertical and 2 for horizontal
#     #import pdb;pdb.set_trace()
#     #min_value = min(cost_matrix[np.shape(cost_matrix)[0]-1,1:])
#     cost_array = cost_matrix[np.shape(cost_matrix)[0]-1,1:]
#     #min_index = np.argmin(cost_matrix[np.shape(cost_matrix)[0]-1,1:])
#     #indices = np.where(cost_array==min_value)
#     max_length = 0
#     costs = []
#     path_lengths = []

#     for k in range(1,len(cost_array)):
#         i, j1 = np.shape(cost_matrix)[0]-1,k
#         path_row, path_column = find_path(i, j1, traceback_indexes)
#         costs.append(cost_array[k]/len(path_row))
#         path_lengths.append(len(path_row))
#         # if(len(path_row)>max_length):
#     	   #  max_length = len(path_row)
#     	   #  p_new = path_row
#     	   #  q_new = path_column
#     return costs, path_lengths, cost_array

#FTraceback function by applying the minimum cost heuristic in the last row.
def traceback(cost_matrix, traceback_indexes):

	 #Traceback indices are 0 for diagnol, 1 for vertical and 2 for horizontal
    #import pdb;pdb.set_trace()
    min_value = min(cost_matrix[np.shape(cost_matrix)[0]-1,1:])
    cost_array = cost_matrix[np.shape(cost_matrix)[0]-1,1:]
    #min_index = np.argmin(cost_matrix[np.shape(cost_matrix)[0]-1,1:])
    indices = np.where(cost_array==min_value)
    max_length = 0
    costs = []
    path_lengths = []

    for k in range(len(indices[0])):
        i, j1 = np.shape(cost_matrix)[0]-1,indices[0][k]+1
        path_row, path_column = find_path(i, j1, traceback_indexes)
        costs.append(min_value/len(path_row))
        path_lengths.append(len(path_row))
 
    return costs, path_lengths, cost_array


'''Find path given the indices of the last row using the traceback indices'''
def find_path(i, j1, traceback_indexes):
	#import pdb;pdb.set_trace()
	p_new, q_new = [i], [j1]
	while(i>0 and j1>0):
		new_index = traceback_indexes[i,j1]
	 	if(new_index == 0):
	 		i = i-1
	 		j1 = j1-1
	 	elif(new_index == 1):
	 		i = i-1
	 	else:
	 		j1 = j1-1
	 	p_new.insert(0,i)
	 	q_new.insert(0,j1)
	return array(p_new), array(q_new)

