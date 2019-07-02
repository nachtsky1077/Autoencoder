import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
from math import sqrt

class dA(object):

    def __init__(self, numpy_rng, theano_rng, input_x, n_visible, n_hidden, W, bhid, bvis):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if theano_rng is None:
            theano_rng = RandomStreams(
                seed=numpy_rng.randint(2 ** 30)
            )

        if W is None:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * sqrt(6. / (n_hidden + n_visible)),
                    high=4 * sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(
                value=initial_W,
                name='W',
                borrow=True
            )
        
        if bhid is None:
            bhid = np.asarray(
                np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        if bvis is None:
            bvis = np.asarray(
                np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input_x is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input_x
        
        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input_val):
        return T.nnet.sigmoid(T.dot(input_val, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input_x, corruption_level):
        return None

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

        return (cost, updates)