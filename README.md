GMM
===

Gaussian Mixture Model Implementation in Pyspark

GMM algorithm models the entire data set as a finite mixture of Gaussian distributions,each parameterized by a mean vector, a covariance matrix  and a mixture weights. Here the probability of each point to belong to each cluster is computed along with the cluster statistics.

This distributed implementation of GMM in pyspark estimates the parameters using the  Expectation-Maximization algorithm and considers only diagonal covariance matrix for each component.

How to Run
==========
There are two ways to run this code.

1. Using the library in your Python program.

  You can train the GMM model by invoking the function GMMModel.trainGMM(data,k,n_iter,ct) where    
  
          data is an RDD(of dense or Sparse Vector),   
          k is the number of components/clusters,   
          n_iter is the number of iterations(default 100),   
          ct is the convergence threshold(default 1e-3).

  To use this library in your program simply download the GMMModel.py and GMMClustering.py 
	and add them as Python files along with your own user code as shown below:
	```
	   wget https://raw.githubusercontent.com/FlytxtRnD/GMM/master/GMMModel.py
	   wget https://raw.githubusercontent.com/FlytxtRnD/GMM/master/GMMClustering.py

	   ./bin/spark-submit --master <master> --py-files GMMModel.py,GMMclustering.py   
  	                       <your-program.py> <input_file> <num_of_clusters>    
	                       [--n_iter <num_of_iterations>] [--ct <convergence_threshold>]
	 ```      
	 The returned object "model" has the following attributes **model.Means,model.Covars,model.Weights**.
	 To get the cluster labels and responsibilty matrix(membership values):
	 
	        responsibility_matrix,cluster_labels = GMMModel.resultPredict(model, data)
	        
2.  Running the example GMM clustering script.

    If you'd like to run our example program directly, also download the PyGMM.py file and invoke it
    with spark-submit.
  ```
       wget https://raw.githubusercontent.com/FlytxtRnD/GMM/master/PyGMM.py
      ./bin/spark-submit --master <master> --py-files GMMModel.py,GMMclustering.py
		                  PyGMM.py <input_file> <num_of_clusters>
		                  [--n_iter <num_of_iterations>] [--ct <convergence_threshold>]
	```
	where master is your spark master URL and input file should contain comma separated numeric values.   
	Make sure you enter the full path to the downloaded files.
