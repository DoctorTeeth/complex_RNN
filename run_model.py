#!/usr/bin/env python

import cPickle
import gzip
import theano
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *
import argparse, timeit

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate,
         savefile, scale_penalty, use_scale,
         model, loss_function):

    theano.config.optimizer='None'
    theano.config.exception_verbosity='high'

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # --- Set data params ----------------
    n_input = 2
    n_output = 1

    # --- Manage data --------------------
    n_train = 1e5
    n_test = 1e4
    num_batches = n_train / n_batch

    train_x = np.asarray(np.zeros((time_steps, n_train, 2)),
                         dtype=theano.config.floatX)

    train_y = np.asarray(np.zeros((time_steps, n_train, 1)),
                         dtype=theano.config.floatX)


    train_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                  high=1.,
                                                  size=(time_steps, n_train)),
                                dtype=theano.config.floatX)

    inds = np.asarray(np.random.randint(time_steps/2, size=(train_x.shape[1],2)))
    inds[:, 1] += time_steps/2

    for i in range(train_x.shape[1]):
        train_x[inds[i, 0], i, 1] = 1.0
        train_x[inds[i, 1], i, 1] = 1.0

    train_y_last = (train_x[:,:,0] * train_x[:,:,1]).sum(axis=0)
    train_y_last = np.reshape(train_y_last, (n_train, 1))
    train_y[-1] = train_y_last

    test_x = np.asarray(np.zeros((time_steps, n_test, 2)),
                        dtype=theano.config.floatX)

    test_y = np.asarray(np.zeros((time_steps, n_test, 1)),
                        dtype=theano.config.floatX)


    test_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                 high=1.,
                                                 size=(time_steps, n_test)),
                                dtype=theano.config.floatX)

    inds = np.asarray([np.random.choice(time_steps, 2, replace=False)
                       for i in xrange(test_x.shape[1])])
    for i in range(test_x.shape[1]):
        test_x[inds[i, 0], i, 1] = 1.0
        test_x[inds[i, 1], i, 1] = 1.0

    test_y_last = (test_x[:,:,0] * test_x[:,:,1]).sum(axis=0)
    test_y_last = np.reshape(test_y_last, (n_test, 1))
    test_y[-1] = test_y_last


    # so train_x is (10,100k,2)
    # an train_y is (100k,1)
    # we'd like to add another dimension to the front of y's shape for both
    # train and test, so that train_y is (10,)


    #######################################################################

    # W_params = [theta]

    inputs, parameters, costs = complex_RNN(n_input,
                                            n_hidden,
                                            n_output,
                                            scale_penalty,
                                            rng,
                                            loss_function=loss_function)
    if use_scale is False:
        parameters.pop()

    # TODO: complex_RNN will return outputs instead of costs
    # then here we will do costs = func(outputs)
    # then we can eliminate a lot of arg passing

    gradients = T.grad(costs[0], parameters)

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}

    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], costs[1], givens=givens_test)

    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):

        if (n_iter % int(num_batches) == 0):
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[inds,:])

        mse = train(i % int(num_batches))
        train_loss.append(mse)
        print "Iteration:", i
        print "mse:", mse
        print

        if (i % 50==0):
            mse = test()
            print
            print "TEST"
            print "mse:", mse
            print
            test_loss.append(mse)

            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_test_loss = mse

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

if __name__=="__main__":


    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_batch", type=int, default=20)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--time_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--savefile", required=True)
    parser.add_argument("--scale_penalty", type=float, default=5)
    parser.add_argument("--use_scale", default=True)
    parser.add_argument("--model", default='complex_RNN')
    parser.add_argument("--loss_function", default='MSE')

    args = parser.parse_args()
    arg_dict = vars(args)

    kwargs = {'n_iter': arg_dict['n_iter'],
              'n_batch': arg_dict['n_batch'],
              'n_hidden': arg_dict['n_hidden'],
              'time_steps': arg_dict['time_steps'],
              'learning_rate': np.float32(arg_dict['learning_rate']),
              'savefile': arg_dict['savefile'],
              'scale_penalty': arg_dict['scale_penalty'],
              'use_scale': arg_dict['use_scale'],
              'model': arg_dict['model'],
              'loss_function': arg_dict['loss_function']}

    main(**kwargs)
