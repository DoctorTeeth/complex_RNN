import theano, cPickle
import theano.tensor as T
import numpy as np
import utils as ut


def complex_RNN(n_input, n_hidden, n_output, scale_penalty, rng,
                activate,
                mask,
                inputs,
                W_params,
                index_permute,
                loss_function='CE'):

    [x,y] = inputs
    [theta, scale] = W_params #TODO: don't do this - but that happens
    # automatically when we do W_ops outside the model


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
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)),
                                   dtype=theano.config.floatX),
                        name='h_0')


    # parameters = [V_re, V_im, U, hidden_bias, reflection, out_bias, theta, h_0, scale]
    W_params = [theta, scale]
    parameters = [V_re, V_im, U, hidden_bias, out_bias, h_0]
    parameters += W_params


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
    def recurrence(x_t, h_prev, rnno_prev, V_re, V_im, hidden_bias, out_bias, U):
        # TODO: there must be a way to tell it we don't use cost_prev during the calculation

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

        lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)

        RNN_output = activate(lin_output)

        return h_t, RNN_output

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [V_re, V_im, hidden_bias, out_bias, U]
    sequences = [x]
    # TODO: can we get rid of "hidden states" and "updates"?
    [hidden_states, rnn_outs], updates = theano.scan(fn=recurrence,
                                                       sequences=sequences,
                                                       non_sequences=non_sequences,
                                                       outputs_info=[h_0_batch, T.zeros_like(y[0])] )


    # TODO: can we get rid of cost_prev?
    def cost_fn(rnn_out, y_t, cost_prev):
        if loss_function == 'CE':
            cost_t = T.nnet.categorical_crossentropy(rnn_out, y_t).mean()
        elif loss_function == 'MSE':
            cost_t = ((rnn_out - y_t)**2).mean()

        return cost_t

    cost_steps, upd = theano.scan(fn=cost_fn,
                                    sequences=[rnn_outs, y],
                                    outputs_info=[ theano.shared(np.float64(0.0)) ] )

    cost = mask(cost_steps)

    cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()
    costs = [cost_penalty, cost]

    # TODO: we should return outputs instead of costs
    return parameters, costs
