import numpy as np
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
import tensorflow as tf
from dualing.datasets import BalancedPairDataset
from dualing.models import ContrastiveSiamese
from dualing.models.base import MLP
from utils.similarity import compute_similarity

# Loading a .txt file to a numpy array
txt = l.load_txt('data/semeion.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=0.8, random_state=1)

# Creates the training and validation datasets
train = BalancedPairDataset(X_train, Y_train, n_pairs=1000, batch_size=128, input_shape=(X_train.shape[0], 256), normalize=(0, 1))
val = BalancedPairDataset(X_val, Y_val, n_pairs=100, batch_size=128, input_shape=(X_val.shape[0], 256), normalize=(0, 1))

# Creates the base architecture
mlp = MLP(n_hidden=[256, 128, 64])

# Creates the contrastive siamese network
s = ContrastiveSiamese(mlp, name='contrastive_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=50)

# Evaluates the network
s.evaluate(val.batches)

#
compute_similarity(s, train.preprocess(X_train))
