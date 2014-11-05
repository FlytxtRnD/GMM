"""
Gaussian Mixture Model
This implementation of GMM in pyspark uses the  Expectation-Maximization algorithm
to estimate the parameters.
"""
import sys
import numpy as np
from pyspark import SparkContext
from GMMModel import GMMModel
from matplotlib import pyplot as plt


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
    if len(sys.argv) < 4:
        print >> sys.stderr, (
            "Invalid number of arguments ::  Usage: "
            "GMM <master> <input_file> <num_of_clusters>[<num_of_iterations>]")
        exit(-1)
    if not(sys.argv[1].startswith('spark')):
        print >> sys.stderr, \
            "Enter spark master URL starting with spark://"
        exit(-1)
    sc = SparkContext(sys.argv[1], appName="GMM",
                      pyFiles=['GMMclustering.py', 'GMMModel.py'])

    input_file = sys.argv[2]
    lines = sc.textFile(input_file)
    data = lines.map(parseVector).cache()
    k = int(sys.argv[3])
    if(len(sys.argv) == 5):
        n_iter = int(sys.argv[4])
    model = GMMModel.trainGMM(data, k, n_iter=100)
    responsibility_matrix,cluster_labels = GMMModel.resultPredict(model,data)

    print cluster_labels.collect()
    # # Writing the GMM components to files
    # means_file = input_file.split(".")[0]+"/means"
    # sc.parallelize(model.Means, 1).saveAsTextFile(means_file)

    # covar_file = input_file.split(".")[0]+"/covars"
    # sc.parallelize(model.Covars, 1).saveAsTextFile(covar_file)

    # responsbilities = input_file.split(".")[0]+"/responsbilities"
    # responsibility_matrix.coalesce(1).saveAsTextFile(responsbilities)

    # cluster_file = input_file.split(".")[0]+"/clusters"
    # cluster_labels.coalesce(1).saveAsTextFile(cluster_file)

    x = data.map(lambda d: ([d[0]])).collect()
    y = data.map(lambda d: ([d[1]])).collect()
    plt.scatter(x, y, c=cluster_labels.collect())
    plt.show()
    matplotlib.pyplot.close()
    sc.stop()
