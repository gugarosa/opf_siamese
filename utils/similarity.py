import tensorflow as tf
import numpy as np


def compute_similarity(model, data):
    """
    """

    #
    n_samples = len(data)

    #
    preds = tf.zeros([0, n_samples])

    #
    for x in data:
        #
        x1 = tf.repeat(tf.expand_dims(x, 0), n_samples, 0)

        #
        x2 = data

        #
        preds = tf.concat([preds, tf.expand_dims(model.predict(x1, x2), 0)], 0)

    #
    np.savetxt('data/semeion_distances.txt', preds.numpy())