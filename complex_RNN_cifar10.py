import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *


# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile,
         scale_penalty, use_scale, reload_progress, model, n_hidden_lstm):

    np.random.seed(1234)
    # --- Set data params ----------------
    n_input = 3
    n_output = 10

    ###### CIFAR preprocessing ##########################################
    # load all data
    for i in xrange(5):
        filename = '/data/lisa/data/cifar10/cifar-10-batches-py/data_batch_%d' % (i+1) 
        f = open(filename, 'rb')
        dict = cPickle.load(f)
        f.close()
        if i==0:
            train_x = dict['data']
            train_y = dict['labels']
        else:
            train_x = np.concatenate((train_x, dict['data']))
            train_y = np.concatenate((train_y, dict['labels']))
                    
    filename = '/data/lisa/data/cifar10/cifar-10-batches-py/test_batch' 
    f = open(filename, 'rb')
    dict = cPickle.load(f)
    f.close()

    test_x = dict['data']
    test_y = dict['labels']
            
    # reshape x data
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] / 3, 3), order='F')
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] / 3, 3), order='F')
    
    train_x = np.swapaxes(train_x, 0, 1)
    test_x = np.swapaxes(test_x, 0, 1)

    n_data = train_x.shape[1]
    num_batches = n_data / n_batch
    n_data_test = test_x.shape[1]

    # change y to one-hot encoding
    temp = np.zeros((n_data, n_output))
    temp[np.arange(n_data), train_y] = 1
    train_y = temp.astype('float32')

    temp = np.zeros((n_data_test, n_output))
    temp[np.arange(n_data_test), test_y] = 1
    test_y = temp.astype('float32')

    # shuffle training data order
    inds = range(n_data)
    np.random.shuffle(inds)
    train_x = train_x[:, inds, :]
    train_y = train_y[inds]

    # rescale pixel values to [0,1]
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    # --- Compile theano graph and gradients-----------------------------------------
 
    gradient_clipping = np.float32(1)
    if (model == 'LSTM'):   
        inputs, parameters, costs = LSTM(n_input, n_hidden, n_output)
    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty)
    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty)
    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output)
    elif (model == 'RNN'):
        inputs, parameters, costs = RNN(n_input, n_hidden, n_output)
    else:
        print "Unsupported model:", model
        return
  
    gradients = T.grad(costs[0], parameters)

#   GRADIENT CLIPPING
    gradients = gradients[:7] + [T.clip(g, -gradient_clipping, gradient_clipping)
            for g in gradients[7:]]
    
#    gradients = clipped_gradients(gradients, gradient_clipping)
 
    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')  

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    train = theano.function([index], [costs[0], costs[2]], givens=givens, updates=updates)
    test = theano.function([], [costs[1], costs[2]], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):
     #   pdb.set_trace()

        [cross_entropy, acc] = train(i % num_batches)
        train_loss.append(cross_entropy)
        print "Iteration:", i
        print "cross_entropy:", cross_entropy
        print "accurracy", acc * 100
        print

        if (i % 100==0):
            [test_cross_entropy, test_acc] = test()
            print
            print "VALIDATION"
            print "cross_entropy:", test_cross_entropy
            print "accurracy", test_acc * 100
            print 
            test_loss.append(test_cross_entropy)
            test_accuracy.append(test_acc)

            if test_cross_entropy < best_test_loss:
                print "NEW BEST!"
                best_params = [p.get_value() for p in parameters]
                best_test_loss = test_cross_entropy

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'test_acc': test_accuracy,
                         'best_test_loss': best_test_loss}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)





    
if __name__=="__main__":
    kwargs = {'n_iter': 100000,
              'n_batch': 20,
              'n_hidden': 512,
              'time_steps': 32*32,
              'learning_rate': np.float32(0.0005),
              'savefile': '/data/lisatmp3/shahamar/complex_RNN/2015-11-10-complexRNN-cifar10.pkl',
              'scale_penalty': 5,
              'use_scale': True,
              'reload_progress': True,
              'model': 'complex_RNN',
              'n_hidden_lstm': 200}
    main(**kwargs)
