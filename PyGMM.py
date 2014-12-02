#
# Copyright 2014 Flytxt
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
    input_file : path of the file which contains the comma separated integer data points
    n_components : Number of mixture components
    n_iter : Number of EM iterations to perform. Default to 100
    ct : convergence_threshold.Default to 1e-3
    """

    conf = SparkConf().setAppName("GMM")
    sc = SparkContext(conf=conf)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('n_components', type=int, help='num_of_clusters')
    parser.add_argument('--n_iter', default=100, type=int, help='num_of_iterations')
    parser.add_argument('--ct', type=float, default=1e-3, help='convergence_threshold')
    args = parser.parse_args()

    input_file = args.input_file
    lines = sc.textFile(input_file)
    data = lines.map(parseVector).cache()

    model = GMMModel.trainGMM(data, args.n_components, args.n_iter, args.ct)
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
