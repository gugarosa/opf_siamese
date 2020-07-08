import argparse

import tensorflow as tf
from dualing.datasets import BalancedPairDataset
from dualing.models import ContrastiveSiamese
from dualing.models.base import MLP

import utils.loader as l
import utils.similarity as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Learns a similarity measure through Siamese Networks.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['boat'])

    parser.add_argument('n_pairs', help='Number of data pairs', type=int)

    parser.add_argument('-batch_size', help='Size of batches', type=int, default=32)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('-epochs', help='Epochs', type=float, default=100)

    return parser.parse_args()

if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    input_file = f'data/{args.dataset}.txt'
    output_file = f'data/{args.dataset}_sim.txt'
    n_pairs = args.n_pairs
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    # Loads the dataset
    X, Y = l.load_dataset(input_file, train_split=None)

    # Creates the dataset
    dataset = BalancedPairDataset(X, Y, n_pairs=n_pairs, batch_size=batch_size,
                                  input_shape=(X.shape[0], X.shape[1]), normalize=(0, 1))

    # Creates the base architecture
    mlp = MLP(n_hidden=[64, 32, 16])

    # Creates the siamese network
    model = ContrastiveSiamese(mlp, name='contrastive_siamese')

    # Compiles the network
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr))

    # Fits the network
    model.fit(dataset.batches, epochs=epochs)

    # Computes the similarity between all samples and saves in a file
    s.compute_similarity(model, dataset.preprocess(X), output_file)
