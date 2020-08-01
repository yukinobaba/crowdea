from pathlib import Path

import numpy as np
import theano
import theano.tensor as T

import pandas as pd
import time

from collections import OrderedDict

def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        name=name,
        borrow=True
    )

class Optimizer(object):
    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]

class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6, params=None):
        super(AdaDelta, self).__init__(params=params)

        self.rho = rho
        self.eps = eps
        self.accugrads = [build_shared_zeros(t.shape.eval(),'accugrad') for t in self.params]
        self.accudeltas = [build_shared_zeros(t.shape.eval(),'accudelta') for t in self.params]

    def updates(self, loss=None):
        super(AdaDelta, self).updates(loss=loss)

        for accugrad, accudelta, param, gparam\
        in zip(self.accugrads, self.accudeltas, self.params, self.gparams):
            agrad = self.rho * accugrad + (1 - self.rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self.eps)/(agrad + self.eps)) * gparam
            self.updates[accudelta] = (self.rho*accudelta + (1 - self.rho) * dx * dx)
            self.updates[param] = param + dx
            self.updates[accugrad] = agrad

        return self.updates

def load_data(data_dir):
    comparison_df = pd.read_csv(Path(data_dir) / 'label.tsv', index_col=0, sep='\t')
    comparison_df = comparison_df[['winner_index', 'loser_index', 'evaluator_index']]

    N = max(comparison_df.winner_index.tolist() + comparison_df.loser_index.tolist()) + 1
    K = max(comparison_df.evaluator_index) + 1
    M = comparison_df.shape[0]

    print('Num. items: {0}, Num. evaluators: {1}, Num. labels: {2}'.format(N,K,M))

    OBS = comparison_df.values
    C = np.zeros((M, N))

    for m in range(M):
        i, j, k = OBS[m, :]

        C[m, i] = 1
        C[m, j] = -1

    C = np.asarray(C)

    return C, N, K, M

def model(N, M_, beta, x_init):
    C = T.matrix(name="C")
    x = theano.shared(x_init, name="x")

    x_diff = T.dot(C, x)
    cost = T.log(1+T.exp(-x_diff)).sum() / M_ + beta * (x ** 2).sum() / N

    params = [x]
    updates = AdaDelta(params=params).updates(cost)

    print('start: compile model')

    train = theano.function(
              inputs=[C],
              outputs=[cost,x],
              updates=updates)

    print('complete: compile model')

    return train

def main(beta, data_dir, result_dir):
    seed = 777
    rs = np.random.RandomState(seed)

    C, N, K, M = load_data(data_dir)

    x_init = rs.randn(N)

    training_epochs = 10
    batch_size = 128
    iter_per_epoch = max(int(M / batch_size), 1)

    train = model(N, batch_size, beta, x_init)

    suffix = 'lambda{}_init'.format(beta)
    np.savetxt(Path(result_dir) / ('bt_x_{0}.dat'.format(suffix)), x_init)

    for i in range(training_epochs):
        start = time.time()
        batch_masks = np.arange(M)
        rs.shuffle(batch_masks)
        costs = []

        for j in range(iter_per_epoch):
            batch_mask = batch_masks[j*batch_size:(j+1)*batch_size]
            cost, x = train(C[batch_mask, :])
            costs.append(cost)

        print('{}: cost:{}'.format(i, np.mean(costs)))
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        start = time.time()

        suffix = 'lambda{}'.format(beta)
        np.savetxt(Path(result_dir) / ('bt_x_{0}.dat'.format(suffix)), x)

        if i % 50 == 0:
            suffix = 'lambda{}_epoch{}'.format(beta, i)
            np.savetxt(Path(result_dir) / ('bt_x_{0}.dat'.format(suffix)), x)

if __name__ == '__main__':
    beta = 0.01

    data_cat = 'idea'
    data_key = 'cheat'

    data_dir = Path('../data/') / data_cat / data_key
    result_dir = Path('../result/') / data_cat / data_key
    result_dir.mkdir(parents=True, exist_ok=True)

    main(beta, data_dir, result_dir)
