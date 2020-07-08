import numpy as np
import tensorflow as tf


def compute_similarity(model, data, file_path):
    """Computes the similarity between all samples.

    Args:
        model (Siamese): A Siamese Network architecture.
        data (np.array): Samples to have their similarity calculated.
        file_path (str): Output file to save the similarities.

    """

    # Gathers the number of samples
    n_samples = len(data)

    # Creates an empty tensor
    preds = tf.zeros([0, n_samples])

    # For every possible sample
    for x in data:
        # Repeats the sample for number of available samples
        x1 = tf.repeat(tf.expand_dims(x, 0), n_samples, 0)

        # Gathers every sample
        x2 = data

        # Performs a prediction using the network and concatenates with the output tensor
        preds = tf.concat([preds, tf.expand_dims(model.predict(x1, x2), 0)], 0)

    # Saves the numpy array to a file
    np.savetxt(file_path, preds.numpy())
