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
            if param.name == 'x':
                x = param + dx
                x *= T.cast(x >= 0, 'float32')
                self.updates[param] = x

            elif param.name == 'w' or param.name == 'v':
                x = param + dx
                x *= T.cast(x >= 0, 'float32')
                x = x / np.sqrt((x ** 2).sum(axis=1, keepdims=True))
                self.updates[param] = x

            else:
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
    R = np.zeros((M, K))

    for m in range(M):
        i, j, k = OBS[m, :]

        C[m, i] = 1
        C[m, j] = -1
        R[m, k] = 1

    C = np.asarray(C)
    R = np.asarray(R)

    P = []
    Q = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            p = np.zeros(N)
            p[i] = 1
            p[j] = -1
            P.append(p)

            q = np.zeros(N)
            q[i] = 1
            Q.append(q)

    P = np.asarray(P)
    Q = np.asarray(Q)
    NN = P.shape[0]

    INPUT = (C, R, P, Q)

    return INPUT, N, K, M, NN

def model(N, K, M_, NN, alpha, D, x_init, w_init, v_init):
    C = T.matrix(name="C")
    R = T.matrix(name="R")
    P = T.matrix(name="P")
    Q = T.matrix(name="Q")

    x = theano.shared(x_init, name="x")
    w = theano.shared(w_init, name="w")
    v = theano.shared(v_init, name="v")

    w_obs = T.dot(R, w)
    x_diff = T.dot(C, x)
    L_c = T.maximum(((T.ones((M_, M_)) - T.dot(w_obs, x_diff.T)) * T.eye(M_)), T.zeros((M_, M_))).sum() / M_

    v_pair = T.dot(Q, v)
    x_diff = T.dot(P, x)
    L_f = T.maximum(((T.ones((NN, NN)) - T.dot(v_pair, x_diff.T)) * T.eye(NN)), T.zeros((NN, NN))).sum() / NN
    cost = L_c + alpha * L_f

    params = [w, v, x]
    updates = AdaDelta(params=params).updates(cost)

    print('start: compile model')

    train = theano.function(
              inputs=[C,R,P,Q],
              outputs=[L_c,L_f,cost,x,v,w],
              updates=updates)

    print('complete: compile model')

    return train

def main(alpha, D, data_dir, result_dir):
    seed = 777
    rs = np.random.RandomState(seed)

    INPUT, N, K, M, NN = load_data(data_dir)

    x_init = rs.rand(N, D)
    w_init = rs.rand(K, D)
    w_init = w_init / np.sqrt((w_init ** 2).sum(axis=1, keepdims=True))
    v_init = rs.rand(N, D)
    v_init = v_init / np.sqrt((v_init ** 2).sum(axis=1, keepdims=True))

    training_epochs = 500
    batch_size = 128
    iter_per_epoch = max(int(M / batch_size), 1)

    train = model(N, K, batch_size, NN, alpha, D, x_init, w_init, v_init)

    suffix = 'alpha{0}_d{1}_init'.format(alpha, D)
    np.savetxt(Path(result_dir) / ('crowdea_x_{0}.dat'.format(suffix)), x_init)
    np.savetxt(Path(result_dir) / ('crowdea_v_{0}.dat'.format(suffix)), v_init)
    np.savetxt(Path(result_dir) / ('crowdea_w_{0}.dat'.format(suffix)), w_init)

    print('alpha:{0}, d:{1}'.format(alpha, D))

    for i in range(training_epochs):
        start = time.time()
        C, R, P, Q = INPUT
        batch_masks = np.arange(M)
        rs.shuffle(batch_masks)
        costs = []
        L_cs = []
        L_fs = []

        for j in range(iter_per_epoch):
            batch_mask = batch_masks[j*batch_size:(j+1)*batch_size]
            L_c, L_f, cost, x, v, w = train(C[batch_mask, :], R[batch_mask, :], P, Q)
            L_cs.append(L_c)
            L_fs.append(L_f)
            costs.append(cost)

        print('{}: L_c:{}, L_f:{}, cost:{}'.format(i, np.mean(L_cs), np.mean(L_fs), np.mean(costs)))
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        start = time.time()

        suffix = 'alpha{0}_d{1}'.format(alpha, D)
        np.savetxt(Path(result_dir) / ('crowdea_x_{0}.dat'.format(suffix)), x)
        np.savetxt(Path(result_dir) / ('crowdea_v_{0}.dat'.format(suffix)), v)
        np.savetxt(Path(result_dir) / ('crowdea_w_{0}.dat'.format(suffix)), w)

        if i % 50 == 0:
            suffix = 'alpha{0}_d{1}_epoch{2}'.format(alpha, D, i)
            np.savetxt(Path(result_dir) / ('crowdea_x_{0}.dat'.format(suffix)), x)
            np.savetxt(Path(result_dir) / ('crowdea_v_{0}.dat'.format(suffix)), v)
            np.savetxt(Path(result_dir) / ('crowdea_w_{0}.dat'.format(suffix)), w)

if __name__ == '__main__':
    alpha = 0.1
    d = 2

    data_cat = 'idea'
    data_key = 'cheat'

    data_dir = Path('../data/') / data_cat / data_key
    result_dir = Path('../result/') / data_cat / data_key
    result_dir.mkdir(parents=True, exist_ok=True)

    main(alpha, 2, data_dir, result_dir)
