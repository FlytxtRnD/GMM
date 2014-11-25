import numpy as np
from pyspark import SparkContext
from GMMclustering import GMMclustering
from pyspark.mllib.linalg import Vectors, SparseVector


class GMMModel(object):
    """
    A clustering model derived from the Gaussian Mixture model.

    >>> data = sc.parallelize(np.array([0.5,1,0.75,1,-0.75,0.5,-0.5,0.5,\
        -1,-0.5,-0.75,-0.75,0.75,-0.5,0.75,-0.75]).reshape(8,2))
    >>> model = GMMModel.trainGMM(data,4,10)
    >>> np.argmax(model.predict(np.array([0.5,1]))) == \
        np.argmax(model.predict(np.array([0.75,1])))
    True
    >>> np.argmax(model.predict(np.array([-0.75,0.5]))) == \
        np.argmax(model.predict(np.array([-0.5,0.5])))
    True
    >>> np.argmax(model.predict(np.array([-1,-0.5]))) == \
        np.argmax(model.predict(np.array([0.75,-0.5])))
    False
    >>> np.argmax(model.predict(np.array([0.75,-0.75]))) == \
        np.argmax(model.predict(np.array([-0.75,-0.75])))
    False

    >>> sparse_data = ([Vectors.sparse(3, {1: 1.0}),\
                    Vectors.sparse(3, {1: 1.1}),\
                    Vectors.sparse(3, {2: 1.0}),\
                    Vectors.sparse(3, {2: 1.1})])
    >>> sparse_data_rdd = sc.parallelize(sparse_data)
    >>> model = GMMModel.trainGMM(sparse_data_rdd,2,10)
    >>> np.argmax(model.predict(np.array([0., 1., 0.]))) == \
        np.argmax(model.predict(np.array([0, 1.1, 0.])))
    True
    >>> np.argmax(model.predict(Vectors.sparse(3, {1: 1.0}))) == \
        np.argmax(model.predict(Vectors.sparse(3, {2: 1.0})))
    False
    >>> np.argmax(model.predict(sparse_data[2])) == \
        np.argmax(model.predict(sparse_data[3]))
    True
    """

    @classmethod
    def trainGMM(cls, data, k, n_iter=100):
        """
        Train a GMM clustering model.
        """
        gmmObj = GMMclustering().fit(data, k, n_iter)
        return gmmObj

    @classmethod
    def resultPredict(cls, gmmObj, data):
        """
        Get the result of predict
        Return responsibility matrix and cluster labels .
        """
        responsibility_matrix = data.map(lambda m: gmmObj.predict(m))
        cluster_labels = responsibility_matrix.map(lambda b: np.argmax(b))
        return responsibility_matrix, cluster_labels


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
