
import numpy as np
from pyspark import SparkContext
from GMMclustering import GMMclustering
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib._common import \
    _get_unmangled_rdd, _get_unmangled_double_vector_rdd

class GMMModel(object):
    """
    A clustering model derived from the Gaussian Mixture model.

    >>> data = np.array([0.3,0.8,-0.58,0.75,-1,-0.40 ,0.70,-0.20,0.25,0.50,\
        -0.85,0.45,-0.55,-0.75,0.75,-1]).reshape(8,2)
    >>> model = GMMModel.trainGMM(sc.parallelize(data),4,10)
    >>> np.argmax(model.predict(np.array([0.3,0.8]))) == \
        np.argmax(model.predict(np.array([0.25,0.50])))
    True
    >>> np.argmax(model.predict(np.array([-0.58,0.75]))) == \
        np.argmax(model.predict(np.array([-0.85,0.45])))
    True
    >>> np.argmax(model.predict(np.array([-1,-0.40]))) == \
        np.argmax(model.predict(np.array([0.70,-0.20])))
    False
    >>> np.argmax(model.predict(np.array([0.75,-1]))) == \
        np.argmax(model.predict(np.array([-0.85,0.45])))
    False
    >>> sparse_data = [
    ...     SparseVector(3, {1: 1.0}),
    ...     SparseVector(3, {1: 1.1}),
    ...     SparseVector(3, {2: 1.0}),
    ...     SparseVector(3, {2: 1.1})
    ... ]
    >>> sparse_data_rdd = sc.parallelize(sparse_data).map(sparseToArray)
    >>> model = GMMModel.trainGMM(sparse_data_rdd,2,10)
    >>> np.argmax(model.predict(np.array([0., 1., 0.]))) == \
        np.argmax(model.predict(np.array([0, 1.1, 0.])))
    True
    """

    @classmethod
    def trainGMM(cls, data, k, n_iter=100):
        """
        Train a GMM clustering model.
        """
        # data = _get_unmangled_double_vector_rdd(data)
        gmmObj = GMMclustering().fit(data, k, n_iter)
        return gmmObj

    @classmethod
    def resultPredict(cls, gmmObj, data):
        """
        Get the result of predict
        Return responsibility matrix and cluster labels .
        """
        # predictRdd = data.map(lambda y:gmmObj.predict(y)
        # responsibility_matrix = predictRdd.map(lambda c: c[0]).collect()
        # cluster_labels = predictRdd.map(lambda c: c[1]).collect()
        responsibility_matrix = data.map(lambda m: gmmObj.predict(m))
        cluster_labels = responsibility_matrix.map(lambda b: np.argmax(b))
        return responsibility_matrix,cluster_labels

def sparseToArray(sparsevector):
        arr = np.zeros(sparsevector.size) 
        for i in xrange(sparsevector.indices.size): 
            arr[sparsevector.indices[i]] = sparsevector.values[i] 
        return arr 

def _test():
    import doctest
    globs = globals().copy()
    globs['sc'] = SparkContext('local[4]', 'PythonTest', batchSize=2)
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    globs['sc'].stop()
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
