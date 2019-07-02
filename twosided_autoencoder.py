import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
from math import sqrt

class TwoDimAutoEncoder(object):

    def __init__(self, numpy_rng, input_x, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # left mult
        initial_W1 = np.asarray(
            numpy_rng.uniform(
                low=-4 * sqrt(6. / (input_size[0] * input_size[1] + output_size[0] * output_size[1])),
                hight= 4 * sqrt(6. / (input_size[0] * input_size[1] + output_size[0] * output_size[1])),
                size=(output_size[0], input_size[0])
            )
        )
        W1 = theano.shared(
            value=initial_W1,
            name='W1',
            borrow=True
        )

        # right mult
        initial_W2 = np.asarray(
            numpy_rng.uniform(
                low=-4 * sqrt(6. / (input_size[0] * input_size[1] + output_size[0] * output_size[1])),
                hight= 4 * sqrt(6. / (input_size[0] * input_size[1] + output_size[0] * output_size[1])),
                size=(input_size[1], output_size[1])
            )
        )
        W2 = theano.shared(
            value=initial_W2,
            name='W2',
            borrow=True
        )

        bvis = np.asarray(
            np.zeros(
                input_size,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        bhid = np.asarry(
            np.zeros(
                output_size,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W1 = W1
        self.W1_prime = self.W1.T
        self.W2 = W2
        self.W2_prime = self.W2.T
        self.b = bhid
        self.b_prime = bvis

        if input_x is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input_x
        
        self.params = [self.W1, self.W2, self.b, self.b_prime]

    def get_hidden_values(self, input_val):
        return T.nnet.sigmoid(T.dot(T.dot(self.W1, input_val), self.W2) + self.b)
    
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(T.dot(self.W1_prime, hidden), self.W2_prime) + self.b_prime)
    
    def get_cost_updates(self, learning_rate):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)

        L = (self.x - z) ** 2

        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

        return (cost, updates)


if __name__ == '__main__':

    x = T.matrix('x')
    index = T.scalar()
    learning_rate = 1e-3
    batch_size = 128
    
    rng = np.random.RandomState(23455)
    twodA = TwoDimAutoEncoder(
        numpy_rng=rng,
        input_x=x,
        input_size=(28, 28),
        output_size=(4, 4)
    )

    cost, updates = twodA.get_cost_updates(learning_rate)
    train_twodA = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )