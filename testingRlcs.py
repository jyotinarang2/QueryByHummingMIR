import matplotlib.pyplot as plt
import rlcs as rlcs
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

x = [1,1,1,0,0,0,1,1,1,1,1,1,1]
y = [1,0,1]
np_query = np.asarray(y)
np_audio = np.asarray(x)
score, diag, cost = rlcs.rlcs(np_query, np_audio)
segment = rlcs.backtrack(np_query, np_audio, score, diag, cost)
print(score)
print(segment)

path = segment[:-1,0:2]
path_length = path.shape[0]
print(path_length)
#imshow(score)
