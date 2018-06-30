import numpy as np


def LogisticDataset(dim, num = 50,  seed = 666):
    """ generate the dataset for logistic objective 
    X has $num$ instances, and Y is the corresponding label for X. 
    Every instance is drawn from one of two multi-variate Gaussians, with half from each.
    Instances from the same Gaussian are assigned the same label, and instances from different Gaussians are assigned different labels.

    Parameters:
    ----------
    dim: the dimension of the generated data.
    num: number of instances in the generated dataset
    seed: fix the random seed for the numpy.
    
    Returns:
    ----------
    X: the generated dataset
    Y: the corresponding label of data X.
    """

    X = np.zeros((num, dim))

    np.random.seed(seed)
    co_var = np.random.rand(dim,dim) * 2
    co_var = co_var.dot(co_var.T)
    mean = np.random.rand(dim) * 5
    X[:int(num / 2),:] = np.random.multivariate_normal(mean, co_var, size = int(num / 2))

    np.random.seed(seed * 2)
    co_var = np.random.rand(dim,dim) * 2
    co_var = co_var.dot(co_var.T)
    mean = np.random.rand(dim) * 5
    X[int(num / 2):,:] = np.random.multivariate_normal(mean, co_var, size = int(num / 2))

    X = np.concatenate((X, np.ones((num, 1))), axis = 1)

    Y = np.zeros(num)
    Y[:int(num / 2)] = 1

    np.random.seed(seed * 3)
    perm = np.random.permutation(num)
    X = X[perm]
    Y = Y[perm]

    # print(X.shape, Y.shape)
    return X, Y


def NeuralNetDataset(dim, num = 100, seed = 233):
    """ generate the dataset for neural-net objective 
    X has $num$ instances, and Y is the corresponding label for X. 
    Every instance is drawn from one of two multi-variate Gaussians, with half from each.
    Instances from the same Gaussian are assigned the same label, and instances from different Gaussians are assigned different labels.

    Parameters:
    ----------
    dim: the dimension of the generated data.
    num: number of instances in the generated dataset
    seed: fix the random seed for the numpy.
    
    Returns:
    ----------
    X: the generated dataset
    Y: the corresponding label of data X.
    """
    def produce(seed):
        np.random.seed(seed)
        co_var = np.random.rand(dim, dim)
        co_var = co_var.dot(co_var.T)
        mean = np.random.rand(dim)
        return mean, co_var


    X = np.zeros((num, dim))

    size = num // 4

    mean, co_var = produce(seed)
    X[:size,:] = np.random.multivariate_normal(mean, co_var, size=size)

    mean, co_var = produce(seed * 2)
    X[size:2*size,:] = np.random.multivariate_normal(mean, co_var, size=size)

    mean, co_var = produce(seed * 4)
    X[2*size:3*size,:] = np.random.multivariate_normal(mean, co_var, size=size)

    mean, co_var = produce(seed * 8)
    X[3*size:,:] = np.random.multivariate_normal(mean, co_var, size=size)

    Y = np.zeros(num)
    Y[2*size:3*size] = 1

    np.random.seed(seed * 16)
    perm = np.random.permutation(num)
    X = X[perm]
    Y = Y[perm]

    return X, Y


if __name__ == '__main__':
    LogisticDataset(dim = 3)

