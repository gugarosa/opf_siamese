import argparse
import pickle

import opfython.math.distance as d


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.
        
    """

    parser = argparse.ArgumentParser(usage='Processes the report into a text file.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['bbc', 'caltech', 'mpeg7', 'semeion'])

    parser.add_argument('-seed', help='Deterministic seed', type=int, default=0)

    parser.add_argument('--use_similarity', help='Whether pre-computed similarity should be used or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed
    use_similarity = args.use_similarity
    input_file = f'outputs/{dataset}_{use_similarity}_{seed}_report.pkl'
    output_file = f'outputs/{dataset}_{use_similarity}_{seed}_report.txt'

    # Opening the input file
    with open(input_file, 'rb') as f:
        # Loading the reports
        report = pickle.load(f)

    # Opening the output file
    with open(output_file, 'w') as f:
        # Writing the header
        f.write('accuracy,precision,recall,f1\n')

        # Writing the metrics
        f.write(f"{report['accuracy']},{report['macro avg']['precision']},"
                f"{report['macro avg']['recall']},{report['macro avg']['f1-score']}\n")
