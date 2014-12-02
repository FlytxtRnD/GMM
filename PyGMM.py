
"""
Gaussian Mixture Model
This implementation of GMM in pyspark uses the  Expectation-Maximization algorithm
to estimate the parameters.
"""
import sys
import argparse
import numpy as np
from GMMModel import GMMModel
from pyspark import SparkContext, SparkConf


def parseVector(line):
    return np.array([float(x) for x in line.split(',')])

if __name__ == "__main__":
    """
    Parameters
    ----------
    master : spark master URL
    input_file : path of the file which contains the comma separated integer data points
    num_of_clusters : Number of mixture components
    num_of_iterations : Number of EM iterations to perform. Default to 100
    """

    conf = SparkConf().setAppName("GMM")
    sc = SparkContext(conf=conf)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('k', type=int, help='num_of_clusters')
    parser.add_argument('--n_iter', default=100, type=int, help='num_of_iterations')
    parser.add_argument('--ct', type=float, default=1e-3, help='convergence_threshold')
    args = parser.parse_args()

    input_file = args.input_file
    lines = sc.textFile(input_file)
    data = lines.map(parseVector).cache()

    model = GMMModel.trainGMM(data, args.k, args.n_iter, args.ct)
    responsibility_matrix, cluster_labels = GMMModel.resultPredict(model, data)

    # Writing the GMM components to files
    means_file = input_file.split(".")[0]+"/means"
    sc.parallelize(model.Means, 1).saveAsTextFile(means_file)

    covar_file = input_file.split(".")[0]+"/covars"
    sc.parallelize(model.Covars, 1).saveAsTextFile(covar_file)

    responsbilities = input_file.split(".")[0]+"/responsbilities"
    responsibility_matrix.coalesce(1).saveAsTextFile(responsbilities)

    cluster_file = input_file.split(".")[0]+"/clusters"
    cluster_labels.coalesce(1).saveAsTextFile(cluster_file)
    sc.stop()
