# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2016-03-07 18:04:47
# @Last Modified by:   Athul
# @Last Modified time: 2016-04-05 12:53:27

# A python implementation of algorithm in
# 1.Shrey Dutta, Krishnaraj Sekhar PV, Hema A. Murthy:
#   Raga Verification in Carnatic Music Using Longest Common Segment Set. ISMIR 2015: 
#    605-611
# 2.S. Dutta and H. A. Murthy, "A modified rough longest common subsequence algorithm for
#   motif spotting in an Alapana of Carnatic Music," Communications (NCC), 2014 Twentieth
#   National Conference on, Kanpur, 2014, pp. 1-6.
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def distance(x, y, measure="euclidean"): # distance measure.
    if (measure == "euclidean"):
        return np.linalg.norm(x - y)**2
    elif (measure == "cubic"):
        return np.power((x - y), 3)  # TODO - doesn't work for multi dim
    else:
        print("Invalid measure! ")

def backtrack(X, Y, score, diag, cost):
    '''Backtracks through the score matrix produced to find the matching signals.
    Returns a variable segment which is a p x 3 matrix.
        p is the length of subsequence set.
        First column denotes index in query
        Second column denotes index in reference
        Third column denotes corresponding score
    Cut at zeros in scores column to find exact set of subsequences'''
    # Find max and min of score matrix
    maxScore = -1*np.inf
    for i in xrange(score.shape[0]):
        for j in xrange(score.shape[1]):
            if (maxScore < score[i, j]):
                maxScore = score[i, j]
                x, y = i, j         # indices of max score
    segLen = 0
    segment = np.array([[0, 0, 0]], dtype=np.float64)   # stores ref position, query position, color for each point
    while(x != 0) and (y != 0):
        i = segLen
        segment[i][0] = x - 1
        segment[i][1] = y - 1
        segment[i][2] = cost[x, y]
        segment = np.vstack((segment, np.array([0, 0, 0])))  # make row for next point
        segLen += 1
        # go to new point        
        if (diag[x, y] == 1):
            x -= 1
            y -= 1
        elif (diag[x, y] == 2):
            x -=1
        else:
            y -= 1
    segment = segment[:-1]          # One row added but while loop exited. remove that
    segment = np.flipud(segment)    # We were backtracking.
    return segment
    

def rlcs(X, Y, tau_dist=0.005, delta=0.5):
    '''Performs the dynamic programming for RLCS. And finally returns score matrices for further analysis.
    parameters:
        X - Query signal; Either a 1D numpy array or 2D array with columns as feature dim and rows as number of samples.
        Y - Reference signal; Either a 1D numpy array or 2D array with columns as feature dim and rows as number of samples
        tau_dist - with normalized distance below tau_dist, samples are considered similar.
        delta - penalty for gap.
    '''
    m = X.shape[0]   # Expect matrix/ multidimensional input
    n = Y.shape[0]
    # find min distance and max distance
    maxDist = minDist = distance(X[0], Y[0])
    for i in xrange(m):
        for j in xrange(n):
            dist = distance(X[i], Y[j])
            if dist > maxDist:
                maxDist = dist
            if dist < minDist:
                minDist = dist
    # Initialize matrices for dynamic programming
    cost = np.zeros((m+2, n+2), dtype=np.float64) # for storing running score
    score = np.zeros((m+2, n+2), dtype=np.float64) # for storing score
    diag = np.zeros((m+2, n+2), dtype=np.float64) # For backtracking cross = 0, up = 2, left = 3
    partial = np.zeros((m+2, n+2), dtype=np.float64) # for storing partial scores.
    p = min(m, n)

    loop_count = 0
    for i in xrange(1, m+1):
        for j in xrange(1, n+1):
            xi = X[i-1]
            yj = Y[j-1]
            # find the distance
            dist = (distance(xi, yj) - minDist)/(maxDist - minDist)
            # take dp action
            if dist < tau_dist:
                diag[i, j] = 1 # cross arrow
                cost[i, j] = cost[i-1, j-1] + (1 - dist/tau_dist)
                score[i, j] = score[i-1, j-1]
            elif (cost[i-1, j] > cost[i, j-1]):
                diag[i, j] = 2 # up arrow
                cost[i, j] = cost[i-1, j] - delta
                if cost[i, j] < 0:
                    cost[i, j] = 0
                if (diag[i-1, j] == 1):
                    score[i, j] = (partial[i-1, j]*np.square(p) + np.square(cost[i-1, j]))/np.square(p)
                else:
                    score[i, j] = score[i-1, j]
            else:
                diag[i, j] = 3 # left arrow
                cost[i, j] = cost[i, j-1] - delta
                if cost[i, j] < 0:
                    cost[i, j] = 0
                if (diag[i, j-1] == 1):
                    score[i, j] = (partial[i, j-1]*np.square(p) + np.square(cost[i-1, j]))/np.square(p)
                else:
                    score[i, j] = score[i, j-1]

            a = partial[i-1, j-1]
            b = partial[i-1, j]
            c = partial[i, j-1]
            mabc = max(a, b, c)

            if (mabc == -1) and (diag[i, j] == 1):
                partial[i, j] = score[i-1, j-1]
            # TODO Confusion here. Is it 0.5 or delta?
            elif (cost[i, j] <= delta) and (diag[i, j] != 1):
                partial[i, j] = -1
            else:
                partial[i, j] = mabc
            loop_count += 1

    lri = score.shape[0] - 1; # last row index of score matrix
    lci = score.shape[1] - 1; # last column index of score matrix
    for i in xrange (lri):
        # last row
        if (diag[i, lci-1] == 1):
            score[i, lci] = (partial[i, lci-1]*np.square(p) + np.square(cost[i, lci-1]))/np.square(p)
        else:
            score[i, lci] = score[i, lci-1]
        diag[i, lci] = 3                    # left

    for i in xrange (lci):
        # last row
        if (diag[lri-1, i] == 1):
            score[lri, i] = (partial[lri-1, i]*np.square(p) + np.square(cost[lri-1, i]))/np.square(p)
        else:
            score[lri, i] = score[lri-1, i]
        diag[lri, i] = 2                    # up

    score[lri, lci] = score[lri-1, lci]
    diag[lri, lci] = 2
    print(dist)
    return score, diag, cost

def plotLCS(segment, X, Y):
    '''Plots the common subsequence with the score.
    From one sample to another, The following inferences can be drawn.
    1. Diagonal movement - next sample is a match - score increases.
    2. Right - A gap is fount, next sample of query is matched with current sample of reference. - score decreases due to penalty.
    3. Up = A gap is fount, current sample of query is matched with next sample of reference. - score decreases due to penalty.

    matplotlib axis and fig is returned.'''
    fig, ax = plt.subplots(figsize=(14, 12))
    match = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
    for m in segment:
        i, j = int(m[0]), int(m[1])
        match[i-1, j-1] = m[2]
    cax = ax.imshow(match, aspect='auto', origin='lower', interpolation="none")
    ax.grid(True)
    cbar = fig.colorbar(cax)
    return fig, ax

def getSoftSegments(segment, X, Y, score_thres=1e-4, len_thres=10):
    idx = np.where(segment[:, 2] > score_thres)[0]
    xSegs = []
    ySegs = []
    j = 0
    for i in xrange(idx.size-1):
        if (idx[i+1] > idx[i] + 1):
            selIdx = idx[j:i+1]
            if len(selIdx) > len_thres:
                xIdx = [int(k) for k in segment[selIdx, 0]]
                xSegs.append(X[xIdx])
                yIdx = [int(k) for k in segment[selIdx, 1]]
                ySegs.append(Y[yIdx])            
            j = i+1
        elif (i == idx.size-2):
            selIdx = idx[j:i+2]
            if len(selIdx) > len_thres:
                xIdx = [int(k) for k in segment[selIdx, 0]]
                xSegs.append(X[xIdx])
                yIdx = [int(k) for k in segment[selIdx, 1]]
                ySegs.append(Y[yIdx])            
            j = i+1
    return xSegs, ySegs


if __name__ == '__main__':
    print(''' This module is not intended to run from interpreter.
        Instead, call the functions from your main script.
        from lcs import rlcs as rlcs

        score, diag, cost = rlcs.rlcs(X, Y, tau_dist,  delta)
        segment = rlcs.backtrack(X, Y, score, diag, cost)''')