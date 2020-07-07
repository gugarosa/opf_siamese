import numpy as np
import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s

# Loading a .txt file to a numpy array
X = np.loadtxt('data/semeion_emb.txt')

# Creating a file of pre-computed distances
g.pre_compute_distance(X, 'data/semeion_distances.txt', distance='euclidean')
