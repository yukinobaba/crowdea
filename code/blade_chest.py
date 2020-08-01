from pathlib import Path
import numpy as np
import theano
#from theano import sparse
import theano.tensor as T
from optimizer import Optimizer
#import scipy.sparse as sp
import utils

import os
import pandas as pd
import time

import argparse

class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6, params=None):
        super(AdaDelta, self).__init__(params=params)

        self.rho = rho
        self.eps = eps
        self.accugrads = [utils.build_shared_zeros(t.shape.eval(),'accugrad') for t in self.params]
        self.accudeltas = [utils.build_shared_zeros(t.shape.eval(),'accudelta') for t in self.params]

    def updates(self, loss=None):
        super(AdaDelta, self).updates(loss=loss)

        for accugrad, accudelta, param, gparam\
        in zip(self.accugrads, self.accudeltas, self.params, self.gparams):
            agrad = self.rho * accugrad + (1 - self.rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self.eps)/(agrad + self.eps)) * gparam
            self.updates[accudelta] = (self.rho*accudelta + (1 - self.rho) * dx * dx)
            if param.name == 'x' or param.name == 'y':
                x = param + dx
                x *= T.cast(x >= 0, 'float32')
                self.updates[param] = x

            else:
                self.updates[param] = param + dx

            self.updates[accugrad] = agrad

        return self.updates

def load_data(data_dir):
    comparison_df = pd.read_csv(os.path.join(data_dir, 'comparison.csv'))
    comparison_df = comparison_df[['winner_index', 'loser_index', 'lancersNickname_index']]

    N = max(comparison_df.winner_index.tolist() + comparison_df.loser_index.tolist()) + 1
    K = max(comparison_df.lancersNickname_index) + 1
    M = comparison_df.shape[0]

    print('Num. items: {0}, Num. workers: {1}, Num. observations: {2}'.format(N,K,M))

    OBS = comparison_df.values
    C = np.zeros((M, N))
    C_winner = np.zeros((M, N))
    C_loser = np.zeros((M, N))

    for m in range(M):
        i, j, k = OBS[m, :]
        C[m, i] = 1
        C[m, j] = -1
        C_winner[m, i] = 1
        C_loser[m, j] = 1

    C = np.asarray(C)
    C_winner = np.asarray(C_winner)
    C_loser = np.asarray(C_loser)

    return C, C_winner, C_loser, N, K, M

def model(N, K, M_, D, beta, x_init, y_init, b_init):
    C = T.matrix(name="C")
    C_winner = T.matrix(name="C_winner")
    C_loser = T.matrix(name="C_loser")

    x = theano.shared(x_init, name="x")
    y = theano.shared(y_init, name="y")
    b = theano.shared(b_init, name="b")

    x_winner = T.dot(C_winner, x)
    x_loser = T.dot(C_loser, x)
    y_winner = T.dot(C_winner, y)
    y_loser = T.dot(C_loser, y)
    diff_b = T.dot(C, b)

    logit = (x_winner * y_loser).sum(axis=1) - (x_loser * y_winner).sum(axis=1) + diff_b
    cost = T.log(1+T.exp(-logit)).sum() / M_ + beta * ((x-y) ** 2).sum() / N

    params = [x, y, b]
    updates = AdaDelta(params=params).updates(cost)

    print('start: compile model')

    train = theano.function(
              inputs=[C, C_winner, C_loser],
              outputs=[cost, x, y, b],
              updates=updates)

    print('complete: compile model')

    return train

def main(beta, D, data_dir, result_dir):
    seed = 777
    rs = np.random.RandomState(seed)

    C, C_winner, C_loser, N, K, M = load_data(data_dir)

    x_init = rs.rand(N, D)
    y_init = rs.rand(N, D)
    b_init = rs.randn(N)

    training_epochs = 500
    batch_size = 128
    iter_per_epoch = max(int(M / batch_size), 1)

    train = model(N, K, batch_size, D, beta, x_init, y_init, b_init)

    suffix = 'lambda{0}_d{1}_init'.format(beta, D)
    np.savetxt(os.path.join(result_dir, 'blade_chest_x_{0}.dat'.format(suffix)), x_init)
    np.savetxt(os.path.join(result_dir, 'blade_chest_y_{0}.dat'.format(suffix)), y_init)
    np.savetxt(os.path.join(result_dir, 'blade_chest_b_{0}.dat'.format(suffix)), b_init)

    print('D:{}'.format(D))

    for i in range(training_epochs):
        start = time.time()
        batch_masks = np.arange(M)
        rs.shuffle(batch_masks)
        costs = []

        for j in range(iter_per_epoch):
            batch_mask = batch_masks[j*batch_size:(j+1)*batch_size]
            cost, x, y, b = train(C[batch_mask, :], C_winner[batch_mask, :], C_loser[batch_mask, :])
            costs.append(cost)

        print('{}: cost:{}'.format(i, np.mean(costs)))
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        start = time.time()

        suffix = 'lambda{0}_d{1}'.format(beta, D)
        np.savetxt(os.path.join(result_dir, 'blade_chest_x_{0}.dat'.format(suffix)), x)
        np.savetxt(os.path.join(result_dir, 'blade_chest_y_{0}.dat'.format(suffix)), y)
        np.savetxt(os.path.join(result_dir, 'blade_chest_b_{0}.dat'.format(suffix)), b)

        if i % 50 == 0:
            suffix = 'lambda{0}_d{1}_epoch{2}'.format(beta, D, i)
            np.savetxt(os.path.join(result_dir, 'blade_chest_x_{0}.dat'.format(suffix)), x)
            np.savetxt(os.path.join(result_dir, 'blade_chest_y_{0}.dat'.format(suffix)), y)
            np.savetxt(os.path.join(result_dir, 'blade_chest_b_{0}.dat'.format(suffix)), b)

if __name__ == '__main__':
    lambd = 0.01
    D = 2

    data_cat = 'idea'
    data_key = 'cheat'

    data_dir = Path('../data/') / data_cat / data_key
    result_dir = Path('../result/') / data_cat / data_key
    result_dir.mkdir(parents=True, exist_ok=True)

    main(lambd, D, data_dir, result_dir)
