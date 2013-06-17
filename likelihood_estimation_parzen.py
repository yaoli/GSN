#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import numpy
import cPickle, gzip
import time

import theano
from theano import tensor as T
from model import load_mnist

def local_contrast_normalization(patches):
    patches = patches.reshape((patches.shape[0], -1))
    patches -= patches.mean(axis=1)[:,None]

    patches_std = numpy.sqrt((patches**2).mean(axis=1))

    min_divisor = (2*patches_std.min() + patches_std.mean()) / 3
    patches /= numpy.maximum(min_divisor, patches_std).reshape((patches.shape[0],1))

    return patches


def log_mean_exp(a):
    max_ = a.max(1)
    
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    x = T.matrix()
    mu = theano.shared(mu)
    
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    
    E = log_mean_exp(-0.5*(a**2).sum(2))
    
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))
    
    return theano.function([x], E - Z)


def numpy_parzen(x, mu, sigma):
    a = ( x[:, None, :] - mu[None, :, :] ) / sigma
    
    def log_mean(i):
        return i.max(1) + numpy.log(numpy.exp(i - i.max(1)[:, None]).mean(1))
    
    return log_mean(-0.5 * (a**2).sum(2)) - mu.shape[1] * numpy.log(sigma * numpy.sqrt(numpy.pi * 2))


def get_ll(x, parzen, batch_size=10):
    inds = range(x.shape[0])
    
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))
    
    times = []
    lls = []
    for i in range(n_batches):
        begin = time.time()
        ll = parzen(x[inds[i::n_batches]])
        end = time.time()
        
        times.append(end-begin)
        
        lls.extend(ll)
        
        if i % 10 == 0:
            print i, numpy.mean(times), numpy.mean(lls)
    
    return lls


def main(sigma, dataset, sample_path='samples.npy'):
    
    # provide a .npy file where 10k generated samples are saved. 
    filename = sample_path
    
    print 'loading samples from %s'%filename
  
    (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist('.')
    
    samples = numpy.load(filename)
    
    parzen = theano_parzen(samples, sigma)
            
    test_ll = get_ll(test_X, parzen)
    
    print "Mean Log-Likelihood of test set = %.5f" % numpy.mean(test_ll)
    print "Std of Mean Log-Likelihood of test set = %.5f" % (numpy.std(test_ll) / 100)


if __name__ == "__main__":
    # to use it on MNIST: python likelihood_estimation_parzen.py 0.23 MNIST
    main(float(sys.argv[1]), sys.argv[2])
    
