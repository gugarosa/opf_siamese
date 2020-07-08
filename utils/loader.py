import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s


def load_dataset(file_path, train_split=0.5, random_state=1):
    """Loads data from a .txt file, parses it and splits (if necessary) into training and validation sets.

    Args:
        file_path (str): Input file to be loaded.
        train_split (float): Percentage of training set.
        random_state (int): Seed used to provide a deterministic trait.

    Returns:
        Dataset or training and validation sets along their indexes.

    """

    # Loading a .txt file to a numpy array
    txt = l.load_txt(file_path)

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Checks if training split is supplied
    if train_split:
        # Splitting data into training and validation sets with their indexes
        X_train, X_val, Y_train, Y_val, I_train, I_val = s.split_with_index(
            X, Y, percentage=train_split, random_state=random_state)

        return X_train, X_val, Y_train, Y_val, I_train, I_val
    
    return X, Y
