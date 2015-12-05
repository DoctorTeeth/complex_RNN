import theano
import theano.tensor as T
import numpy as np

def rms_prop(learning_rate, parameters, gradients):        
    rmsprop = [theano.shared(1e-3*np.ones_like(p.get_value())) for p in parameters]
    new_rmsprop = [0.9 * vel + 0.1 * (g**2) for vel, g in zip(rmsprop, gradients)]

    updates1 = zip(rmsprop, new_rmsprop)
    updates2 = [(p, p - learning_rate * g / T.sqrt(rms)) for 
                p, g, rms in zip(parameters, gradients, new_rmsprop)]
    updates = updates1 + updates2
    return updates, rmsprop
