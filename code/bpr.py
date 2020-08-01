from pathlib import Path
import numpy as np
import theano
import theano.tensor as T
from optimizer import Optimizer
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
            if param.name == 'x':
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
    R = np.zeros((M, K))

    for m in range(M):
        i, j, k = OBS[m, :]

        C[m, i] = 1
        C[m, j] = -1
        R[m, k] = 1

    INPUT = (C, R)

    return INPUT, N, K, M

def model(N, K, M_, D, beta, x_init, w_init):
    C = T.matrix(name="C")
    R = T.matrix(name="R")

    x = theano.shared(x_init, name="x")
    w = theano.shared(w_init, name="w")

    w_obs = T.dot(R, w)
    x_diff = T.dot(C, x)

    logit = (w_obs*x_diff).sum(axis=1)
    cost = T.log(1+T.exp(-logit)).sum() / M_ + beta * ((x** 2).sum() + (w**2).sum()) / N

    params = [w, x]
    updates = AdaDelta(params=params).updates(cost)

    print('start: compile model')

    train = theano.function(
              inputs=[C,R],
              outputs=[cost,x,w],
              updates=updates)

    print('complete: compile model')

    return train

def main(beta, D, data_dir, result_dir):
    seed = 777
    rs = np.random.RandomState(seed)

    INPUT, N, K, M = load_data(data_dir)

    x_init = rs.rand(N, D)
    w_init = rs.randn(K, D)

    training_epochs = 500
    batch_size = 128
    iter_per_epoch = max(int(M / batch_size), 1)

    train = model(N, K, batch_size, D, beta, x_init, w_init)

    suffix = 'lambda{0}_d{1}_init'.format(beta, D)
    np.savetxt(os.path.join(result_dir, 'bpr_x_{0}.dat'.format(suffix)), x_init)
    np.savetxt(os.path.join(result_dir, 'bpr_w_{0}.dat'.format(suffix)), w_init)

    print('D:{}'.format(D))

    for i in range(training_epochs):
        start = time.time()
        C, R, = INPUT
        batch_masks = np.arange(M)
        rs.shuffle(batch_masks)
        costs = []

        for j in range(iter_per_epoch):
            batch_mask = batch_masks[j*batch_size:(j+1)*batch_size]
            cost, x, w = train(C[batch_mask, :], R[batch_mask, :])
            costs.append(cost)

        print('{}: cost:{}'.format(i, np.mean(costs)))
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        start = time.time()

        suffix = 'lambda{0}_d{1}'.format(beta, D)
        np.savetxt(os.path.join(result_dir, 'bpr_x_{0}.dat'.format(suffix)), x)
        np.savetxt(os.path.join(result_dir, 'bpr_w_{0}.dat'.format(suffix)), w)

        if i % 50 == 0:
            suffix = 'lambda{0}_d{1}_epoch{2}'.format(beta, D, i)
            np.savetxt(os.path.join(result_dir, 'bpr_x_{0}.dat'.format(suffix)), x)
            np.savetxt(os.path.join(result_dir, 'bpr_w_{0}.dat'.format(suffix)), w)

if __name__ == '__main__':
    lambd = 0.01
    D = 2

    data_cat = 'idea'
    data_key = 'cheat'

    data_dir = Path('../data/') / data_cat / data_key
    result_dir = Path('../result/') / data_cat / data_key
    result_dir.mkdir(parents=True, exist_ok=True)

    main(lambd, D, data_dir, result_dir)
