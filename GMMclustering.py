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

import logging
import numpy as np
from operator import add
from scipy.misc import logsumexp
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans


class GMMclustering:
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s')

    def fit(self, X, k, n_iter, convergence_threshold):
        """
        Estimate model parameters with the expectation-maximization
        algorithm.

        Parameters
        ----------
        X - RDD of data points
        k - Number of components
        n_iter - Number of iterations. Default to 100

        Attributes
        ----------

        covariance_type : Type of covariance matrix.
            Supports only diagonal covariance matrix.

        convergence_threshold : Threshold value to check the convergence criteria.
            Defaults to 1e-3

        min_covar : Floor on the diagonal of the covariance matrix to prevent
            overfitting.  Defaults to 1e-3.

        converged : True once converged False otherwise.

        Weights : array of shape (1,  k)
            weights for each mixture component.

        Means : array of shape (k, n_dim)
            Mean parameters for each mixture component.

        Covars : array of shape (k, n_dim)
            Covariance parameters for each mixture component

        """
        sc = X.context
        covariance_type = 'diag'
        converged = False
        self.min_covar = 1e-3

        #  observation statistics
        self.s0 = 0
        self.s1 = 0
        #  To get the no of data points
        n_points = X.count()
        #  To get the no of dimensions
        n_dim = X.first().size

        if (n_points == 0):
            raise ValueError(
                'Dataset cannot be empty')
        if (n_points < k):
            raise ValueError(
                'Not possible to make (%s) components from (%s) datapoints' %
                (k,  n_points))

        # Initialize Covars(diagonal covariance matrix)
        if hasattr(X.first(), 'indices'):
            self.isSparse = 1

            def convert_to_kvPair(eachV):
                g = []
                for i in range(eachV.indices.size):
                    g.append((eachV.indices[i],
                             (eachV.values[i], eachV.values[i]*eachV.values[i])))
                return g

            def computeVariance(x):
                mean = x[1][0]/n_points
                sumSq = x[1][1]/n_points
                return x[0], sumSq - mean*mean

            cov = []
            kvPair = X.flatMap(convert_to_kvPair)
            res = kvPair.reduceByKey(np.add).map(computeVariance)
            cov = Vectors.sparse(n_dim, res.collectAsMap()).toArray() + 1e-3
            self.Covars = np.tile(cov,  (k,  1))

        else:
            self.isSparse = 0
            cov = []
            for i in range(n_dim):
                cov.append(X.map(lambda m: m[i]).variance()+self.min_covar)
            self.Covars = np.tile(cov,  (k,  1))

        # Initialize Means using MLlib KMeans
        self.Means = np.array(KMeans().train(X, k).clusterCenters)
        # Initialize Weights with the value 1/k for each component
        self.Weights = np.tile(1.0 / k, k)
        #  EM algorithm
        # loop until number of iterations  or convergence criteria is satisfied
        for i in range(n_iter):

            logging.info("GMM running iteration %s " % i)
            # broadcasting means,covars and weights
            self.meansBc = sc.broadcast(self.Means)
            self.covarBc = sc.broadcast(self.Covars)
            self.weightBc = sc.broadcast(self.Weights)
            # Expectation Step
            EstepOut = X.map(self.scoreOnePoint)
            # Maximization step
            MstepIn = EstepOut.reduce(lambda (w1, x1, y1, z1), (w2, x2, y2, z2):
                                      (w1+w2, x1+x2,  y1+y2,  z1+z2))
            self.s0 = self.s1
            self.mStep(MstepIn[0], MstepIn[1], MstepIn[2], MstepIn[3])

            #  Check for convergence.
            if i > 0 and abs(self.s1-self.s0) < convergence_threshold:
                converged = True
                logging.info("Converged at iteration %s" % i)
                break

        return self

    def scoreOnePoint(self, x):

        """
        Compute the log likelihood of 'x' being generated under the current model
        Also returns the probability that 'x' is generated by each component of the mixture

        Parameters
        ----------
        x : array of shape (1,  n_dim)
            Corresponds to a single data point.

        Returns
        -------
        log_likelihood_x :Log likelihood  of 'x'
        prob_x : Resposibility  of each cluster for the data point 'x'

        """
        lpr = (self.log_multivariate_normal_density_diag_Nd(x) + np.log(self.Weights))
        log_likelihood_x = logsumexp(lpr)
        prob_x = np.exp(lpr-log_likelihood_x)

        if self.isSparse == 1:
            temp_wt = np.dot(prob_x[:, np.newaxis], x.toArray()[np.newaxis, :])
            sqVec = Vectors.sparse(x.size, x.indices, x.values**2)
            temp_avg = np.dot(prob_x.T[:, np.newaxis], sqVec.toArray()[np.newaxis, :])

        else:
            temp_wt = np.dot(prob_x.T[:, np.newaxis],  x[np.newaxis, :])
            temp_avg = np.dot(prob_x.T[:, np.newaxis], (x*x)[np.newaxis, :])

        return log_likelihood_x, prob_x, temp_wt, temp_avg

    def log_multivariate_normal_density_diag_Nd(self, x):
        """
        Compute Gaussian log-density at x for a diagonal model

        """

        n_features = x.size

        if self.isSparse == 1:
            t = Vectors.sparse(x.size, x.indices, x.values**2).dot((1/self.covarBc.value).T)

        else:
            t = np.dot(x**2, (1/self.covarBc.value).T)

        lpr = -0.5 * (n_features*np.log(2*np.pi) + np.sum(np.log(self.covarBc.value), 1) +
                      np.sum((self.meansBc.value ** 2) / self.covarBc.value, 1)
                      - 2 * x.dot((self.meansBc.value/self.covarBc.value).T) + t)

        return lpr

    def mStep(self, log_sum, prob_sum, weighted_X_sum, weighted_X2_sum):
        """
            Perform the Mstep of the EM algorithm.
            Updates Means, Covars and Weights using observation statistics
        """
        self.s1 = log_sum
        inverse_prob_sum = 1.0 / (prob_sum)
        self.Weights = (prob_sum / (prob_sum.sum()))
        self.Means = (weighted_X_sum * inverse_prob_sum.T[:, np.newaxis])
        self.Covars = ((weighted_X2_sum * inverse_prob_sum.T[:, np.newaxis]) - (self.Means**2)
                       + self.min_covar)

    def predict(self, x):
        """
            Predicts the cluster to which the given instance belongs to
            based on the maximum resposibility.

        Parameters
        ----------
        x : array of shape (1,  n_dim)
            Corresponds to a single data point.

        Returns
        -------
        resposibility_matrix:membership values of x in each cluster component
        """
        if hasattr(x, 'indices'):
            self.isSparse = 1

        else:
            self.isSparse = 0

        lpr = (self.log_multivariate_normal_density_diag_Nd(x) + np.log(self.Weights))
        log_likelihood_x = logsumexp(lpr)
        prob_x = np.exp(lpr-log_likelihood_x)
        resposibility_matrix = np.array(prob_x)
        return resposibility_matrix
