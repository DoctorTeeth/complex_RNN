import theano, cPickle
import theano.tensor as T
import numpy as np
import utils as ut



def complex_RNN(n_input, n_hidden, n_output, scale_penalty, out_every_t=False,
                loss_function='CE'):

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V_re = ut.initialize_matrix(n_input, n_hidden, 'V_re', rng)
    V_im = ut.initialize_matrix(n_input, n_hidden, 'V_im', rng)
    U = ut.initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX),
                                name='hidden_bias')

    reflection = ut.initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = ut.initialize_matrix(3, n_hidden, 'theta', rng)
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)),
                                   dtype=theano.config.floatX),
                        name='h_0')

    scale = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX),
                          name='scale')

    # parameters = [V_re, V_im, U, hidden_bias, reflection, out_bias, theta, h_0, scale]
    parameters = [V_re, V_im, U, hidden_bias, out_bias, theta, h_0, scale]

    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    index_permute = np.random.permutation(n_hidden)

    # specify computation of the hidden-to-hidden transform
    W_ops = [ lambda accum: ut.times_diag(accum, n_hidden, theta[0,:]),
              # lambda accum: times_reflection(accum, n_hidden, reflection[0,:]),
              lambda accum: ut.vec_permutation(accum, n_hidden, index_permute),
              lambda accum: ut.times_diag(accum, n_hidden, theta[1,:]),
              # lambda accum: times_reflection(accum, n_hidden, reflection[1,:]),
              lambda accum: ut.times_diag(accum, n_hidden, theta[2,:]),
              lambda accum: ut.scale_diag(accum, n_hidden, scale)
    ]

    # define the recurrence used by theano.scan - U maps hidden to output
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V_re, V_im, hidden_bias, scale, out_bias, U):
        # TODO: there must be a way to tell it we don't use cost_prev during the calculation
        # TODO: we'll need to make do_fft take more args
        # TODO: once we finish moving steps out of the loop, we can not pass U params to recurrence anymore
        # TODO: may have to append onto a list of steps
        # depends how theano compiler works

        hidden_lin_output = reduce(lambda x,f : f(x), W_ops, h_prev)

        # Compute data linear transform
        data_lin_output_re = T.dot(x_t, V_re)
        data_lin_output_im = T.dot(x_t, V_im)
        data_lin_output = T.concatenate([data_lin_output_re, data_lin_output_im], axis=1)

        # Total linear output
        lin_output = hidden_lin_output + data_lin_output
        lin_output_re = lin_output[:, :n_hidden]
        lin_output_im = lin_output[:, n_hidden:]

        # Apply non-linearity ----------------------------
        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
        rescale = T.maximum(modulus + hidden_bias.dimshuffle('x',0), 0.) / (modulus + 1e-5)
        nonlin_output_re = lin_output_re * rescale
        nonlin_output_im = lin_output_im * rescale

        h_t = T.concatenate([nonlin_output_re,
                             nonlin_output_im], axis=1)
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))

        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V_re, V_im, hidden_bias, scale, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=[h_0_batch, theano.shared(np.float32(0.0)),
                                                                                theano.shared(np.float32(0.0))])

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            costs = [cost_penalty, cost]


    else:
        cost = cost_steps.mean()
        cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()
        accuracy = acc_steps.mean()
        costs = [cost_penalty, cost, accuracy]


    return [x, y], parameters, costs
