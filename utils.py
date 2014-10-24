import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time

cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))

def SaltAndPepper(X, rate=0.3):
    # Salt and pepper noise
    drop = numpy.arange(X.shape[1])
    numpy.random.shuffle(drop)
    sep = int(len(drop)*rate)
    drop = drop[:sep]
    X[:, drop[:sep/2]]=0
    X[:, drop[sep/2:]]=1
    return X

def get_shared_weights(n_in, n_out, interval, name):
    #val = numpy.random.normal(0, sigma_sqr, size=(n_in, n_out))
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_bias(n, name, offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def dropout(IN, rng_theano, p = 0.5):
    noise   =   rng_theano.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    OUT     =   (IN * noise) / cast32(p)
    return OUT

def add_gaussian_noise(IN, rng_theano, std = 1):
    print 'GAUSSIAN NOISE : ', std
    noise   =   rng_theano.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
    OUT     =   IN + noise
    return OUT

def corrupt_input(IN, rng_theano, p = 0.5):
    # salt and pepper? masking?
    noise   =   rng_theano.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
    IN      =   IN * noise
    return IN

def salt_and_pepper(IN, rng_theano, p = 0.2):
    # salt and pepper noise
    print 'DAE uses salt and pepper noise'
    a = rng_theano.binomial(size=IN.shape, n=1,
                              p = 1 - p,
                              dtype='float32')
    b = rng_theano.binomial(size=IN.shape, n=1,
                              p = 0.5,
                              dtype='float32')
    c = T.eq(a,0) * b
    return IN * a + c

def load_mnist_real():
    data = cPickle.load(open('./data/mnist.pkl', 'r'))
    return data

def load_mnist_binary():
    data = cPickle.load(open('./data/mnist.pkl', 'r'))
    data = [list(d) for d in data] 
    data[0][0] = (data[0][0] > 0.5).astype('float32')
    data[1][0] = (data[1][0] > 0.5).astype('float32')
    data[2][0] = (data[2][0] > 0.5).astype('float32')
    data = tuple([tuple(d) for d in data])
    return data
    
def load_tfd(path):
    import scipy.io as io
    data = io.loadmat(os.path.join(path, 'TFD_48x48.mat'))
    X = cast32(data['images'])/cast32(255)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    labels  = data['labs_ex'].flatten()
    labeled = labels != -1
    unlabeled   =   labels == -1  
    train_X =   X[unlabeled]
    valid_X =   X[unlabeled][:100] # Stuf
    test_X  =   X[labeled]

    del data

    return (train_X, labels[unlabeled]), (valid_X, labels[unlabeled][:100]), (test_X, labels[labeled])


