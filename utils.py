import theano
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft

def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)

def do_fft(input, n_hidden):
    fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) / T.sqrt(n_hidden)
    fft_output = fft_output.dimshuffle(0,2,1)
    output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

def do_ifft(input, n_hidden):
    ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output


def scale_diag(input, n_hidden, diag):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    Diag = T.nlinalg.AllocDiag()(diag)
    input_re_times_Diag = T.dot(input_re, Diag)
    input_im_times_Diag = T.dot(input_im, Diag)

    return T.concatenate([input_re_times_Diag, input_im_times_Diag], axis=1)

def times_diag(input, n_hidden, diag):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    Re = T.nlinalg.AllocDiag()(T.cos(diag))
    Im = T.nlinalg.AllocDiag()(T.sin(diag))
    input_re_times_Re = T.dot(input_re, Re)
    input_re_times_Im = T.dot(input_re, Im)
    input_im_times_Re = T.dot(input_im, Re)
    input_im_times_Im = T.dot(input_im, Im)

    return T.concatenate([input_re_times_Re - input_im_times_Im,
                          input_re_times_Im + input_im_times_Re], axis=1)

def vec_permutation(input, n_hidden, index_permute):
    re = input[:, :n_hidden]
    im = input[:, n_hidden:]
    re_permute = re[:, index_permute]
    im_permute = im[:, index_permute]

    return T.concatenate([re_permute, im_permute], axis=1)

def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

    vstarv = (reflect_re**2 + reflect_im**2).sum()
    input_re_reflect = input_re - 2 / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re)
                                                + T.outer(T.dot(input_re, reflect_im), reflect_im)
                                                - T.outer(T.dot(input_im, reflect_im), reflect_re)
                                                + T.outer(T.dot(input_im, reflect_re), reflect_im))
    input_im_reflect = input_im - 2 / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re)
                                                + T.outer(T.dot(input_im, reflect_im), reflect_im)
                                                + T.outer(T.dot(input_re, reflect_im), reflect_re)
                                                - T.outer(T.dot(input_re, reflect_re), reflect_im))

    return T.concatenate([input_re_reflect, input_im_reflect], axis=1)
