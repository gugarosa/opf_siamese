import dualing.utils.projector as proj
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
import tensorflow as tf
from dualing.datasets import BalancedPairDataset
from dualing.models import CrossEntropySiamese
from dualing.models.base import MLP

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
x, y = p.parse_loader(txt)

#
y = y - 1

# Splitting data into training and testing sets
# x, x_val, t, y_val = s.split(x, y, percentage=0.8, random_state=1)

# Creates the training and validation datasets
train = BalancedPairDataset(x, y, n_pairs=1000, batch_size=64, input_shape=(x.shape[0], 2), normalize=(0, 1))
# val = BalancedPairDataset(x_val, y_val, n_pairs=10, batch_size=2, input_shape=(x_val.shape[0], 2), normalize=(0, 1))

# Creates the base architecture
mlp = MLP(n_hidden=[128, 64])

# Creates the cross-entropy siamese network
s = CrossEntropySiamese(mlp, name='cross_entropy_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=100)

# Evaluates the network
# s.evaluate(val.batches)

# Extract embeddings
embeddings = s.extract_embeddings(train.preprocess(x))

# Visualize embeddings
proj.plot_embeddings(embeddings, y)
