GMM
===

Gaussian Mixture Model Implementation in Pyspark

GMM algorithm models the entire data set as a finite mixture of Gaussian distributions,each parameterized by a mean vector, a covariance matrix  and a mixture weights. Here the probability of each point to belong to each cluster is computed along with the cluster statistics.

This distributed implementation of GMM in pyspark estimates the parameters using the  Expectation-Maximization algorithm and considers only diagonal covariance matrix for each component.

How to Run
==========

There are two ways to run this code.

1.Run the file PyGMM using spark-submit with the arguments:

          PyGMM <master> <input_file> <num_of_clusters>[<num_of_iterations>]
          
  where master is the spark master URL and input file should be comma separated numeric values.
  The main function will convert the dataset into an RDD and invoke the trainGMM method of class GMMModel.
  
2.Train the GMM model using a dataset:

          model = GMMModel.trainGMM(data,k,n_iter)
          
  where data is an RDD(of Dense or Sparse Vector) ,k is the number of components/clusters, n_iter is the number of   iterations(by default 100)
  
  The returned object "model" has the following attributes:
            model.Means
            model.Covars
            model.Weights
  
  
  To get the cluster labels and responsbility matrix (membership values):
  
          responsibility_matrix,cluster_labels = GMMModel.resultPredict(model, data)

  
