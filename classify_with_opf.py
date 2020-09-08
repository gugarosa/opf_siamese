import argparse
import pickle

import numpy as np
import opfython.math.general as g
from opfython.models import SupervisedOPF
from sklearn.metrics import classification_report

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Classifies data using Optimum-Path Forest.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['bbc', 'caltech', 'mpeg7', 'semeion'])

    parser.add_argument('-tr_split', help='Training set percentage', type=float, default=0.5)

    parser.add_argument('-seed', help='Deterministic seed', type=int, default=0)

    parser.add_argument('--use_similarity', help='Whether pre-computed similarity should be used or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    input_file = f'data/{dataset}.txt'
    input_sim = f'data/{dataset}_sim.txt'
    split = args.tr_split
    seed = args.seed
    use_similarity = args.use_similarity

    # Loads the training and testing sets along their indexes
    X_train, Y_train, I_train, X_test, Y_test, I_test = l.load_split_dataset(
        input_file, train_split=split, random_state=seed)

    # If similarity should be used
    if use_similarity:
        # Creates a SupervisedOPF with pre-computed distances
        opf = SupervisedOPF(pre_computed_distance=input_sim)

    # If similarity should not be used
    else:
        # Creates a SupervisedOPF without pre-computed distances
        opf = SupervisedOPF(distance='log_squared_euclidean')

    # Fits training data into the classifier
    opf.fit(X_train, Y_train, I_train)

    # Predicts new data
    preds = opf.predict(X_test, I_test)

    # Calculates the confusion matrix
    c_matrix = g.confusion_matrix(Y_test, preds)

    # Calculates the classification report
    report = classification_report(Y_test, preds, output_dict=True)

    # Saves confusion matrix in a .npy file
    np.save(f'outputs/{dataset}_{use_similarity}_{seed}_matrix', c_matrix)

    # Opens file to further save
    with open(f'outputs/{dataset}_{use_similarity}_{seed}_report.pkl', 'wb') as f:
        # Saves report to a .pkl file
        pickle.dump(report, f)
