import tensorflow as tf
import numpy as np
import csv
from config import *
from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.cluster import KMeans as km

def variable_creator(name, shape):
    '''
    Creates trainable tensorflow variables and returns them with the given name and shape.
    :param name: String with the name of the variable.
    :param shape: A list with integers to specify the size of the variable.
    :return: Tensorflow variable
    '''
    with tf.name_scope(name=name):
        intial = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable(name=name, shape=shape, initializer=intial)


def loss(y, x, d, lamb, name):
    '''
    Calculates the sparse loss with given inputs and returns the loss value.
    :param y: The reference matrix
    :param x: The output sparse matrix
    :param d: The sparse dictionary(A matrix with sparsity)
    :param lamb: The sparsity coefficient, basically an integer.
    :param name: The name of the loss function used as a string.
    :return: Value of the loss as an integer.
    '''
    with tf.name_scope(name=name):
        diff_tensor = tf.subtract(y, tf.matmul(d, x))
        first_term = tf.norm(diff_tensor, name='First_Term')
        second_term = lamb * tf.norm(x, ord=1, name='Second_Term')
        return first_term + second_term


def normalize_input(data, name):
    '''
    Normalizes the given input matrix and returns the matrix that has normalized values.
    :param data: Input matrix with integers that has to be normalized.
    :param name: Name string for the scope in tensorflow graph.
    :return: Normalized matrix, max value in the original input matrix and least value in the original input matrix.
    '''
    with tf.name_scope(name=name):
        normalized = np.transpose(data).astype(float)
        # max_min = normalized[np.where(normalized != 0.0)]
        for i in range(len(normalized)):
            normalized[i] = (normalized[i]) / (np.max(normalized) - np.min(normalized))
        return np.transpose(normalized), np.max(normalized), np.min(normalized)


def de_normalized(output, max, min):
    '''
    Removes the normalization effect on the output matrix. Uses the max and min values from the original input to
    carry out this operation.
    :param output: Output matrix after training.
    :param max: Maximum value of in the input matrix.
    :param min: Minimum value in the input matrix.
    :return: De-normalized output matrix.
    '''
    output = np.round(output, 2)
    return(output * (max - min))


def input_generator(file_names, n_clust, n_sub, cluster_method, sub_length):
    '''
    Creates a matrix from the gaze data. Gaze data from each file is clustered to n_clust points using either HR
    clustering or KM clustering. These n_clust points are flattened to 2 * n_clust values and each of these 2 * n_clust
    values are divided in n_sub subsequences of length sub_length. The columns of the matrix correspond to these gaze
    subsequences of length sub_length flattened into 2 * sub_length values.
    :param file_names: Gaze files names as the list of strings.
    :param n_clust: Number of clusters as an integer.
    :param n_sub: Number of Sub seqeunces per image as an integer
    :param cluster_method: Clustering method used as a String. Can be either 'HR' or 'KM'
    :param sub_length:
    :return:
    '''
    mat = []
    for image in file_names:
        gaze = []
        with open(args.gaze_path + image, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            count = 0
            for row in reader:
                gaze.append((int(row[0]), int(row[1]), count * 110))
                count += 1
            gaze = np.array(gaze)
            try:
                if cluster_method == 'HR':
                    cluster_labels = hc(n_clusters=n_clust).fit_predict(gaze)
                elif cluster_method == 'KM':
                    cluster_labels = km(n_clusters=n_clust).fit_predict(gaze)
            except:
                raise ValueError('Choose between "HR" or "KM" as the clustering method')
            result = {i: gaze[np.where(cluster_labels == i)] for i in range(n_clust)}
            centres = []
            for cluster in result:
                cluster_points = np.array(result[cluster])
                cluster_centre = np.mean(cluster_points, axis=0)
                centres.append(cluster_centre[0])
                centres.append(cluster_centre[1])
            for i in range(0, len(centres) - (2 * sub_length), 2 * ((n_clust - sub_length)/ n_sub)):
                mat.append(centres[i:i + (2 * sub_length)])
    mat = np.array(mat)
    return np.transpose(mat)
