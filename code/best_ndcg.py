from pathlib import Path
import pandas as pd
import numpy as np
import collections
import scipy.stats

import argparse

import warnings
warnings.filterwarnings('ignore')

from ranking import RankingMeasures

def get_best_ndcg(x, p_truth):
    e = 1
    degrees = np.arange(0, 90.+e, e)

    ndcg5_best = -1
    ndcg10_best = -1
    d = x.shape[1]
    assert(d == 2 or d == 3)

    if d == 2:
        n = 100
    else:
        n = 1000

    if d == 2:
        for r in degrees:
            v_tmp = np.array([np.cos(np.radians(r)), np.sin(np.radians(r))])
            p = x.dot(v_tmp).flatten()
            rm = RankingMeasures(p, p_truth)
            ndcg5 = rm.nDCG(5)
            ndcg10 = rm.nDCG(10)

            if ndcg5 > ndcg5_best:
                ndcg5_best = ndcg5

            if ndcg10 > ndcg10_best:
                ndcg10_best = ndcg10

    else:
        for r in degrees:
            for r_ in degrees:
                v_tmp = np.array([np.sin(np.radians(r))*np.cos(np.radians(r_)),
                                                  np.sin(np.radians(r))*np.sin(np.radians(r_)),
                                                  np.cos(np.radians(r))])
                p = x.dot(v_tmp).flatten()
                rm = RankingMeasures(p, p_truth)
                ndcg5 = rm.nDCG(5)
                ndcg10 = rm.nDCG(10)

                if ndcg5 > ndcg5_best:
                    ndcg5_best = ndcg5

                if ndcg10 > ndcg10_best:
                    ndcg10_best = ndcg10

    return ndcg5_best, ndcg10_best

def main(data_dir_name, trial):
    data_dir = Path('../%s' % data_dir_name)
    result_dir = Path('../%s' % (data_dir_name.replace('data', 'result')))

    data_cats = ['idea', 'design']
    data_keys = {'idea': ['bike', 'cheat', 'meeting', 'night', 'visitor'],
                 'design': ['ai_character', 'olympic']}
    data_keys_arr = data_keys['idea'] + data_keys['design']
    d_arr = [2, 3]
    methods = ['crowdea_%d' % d for d in d_arr]
    methods += ['blade_chest_%d_x' % d for d in d_arr] + ['blade_chest_%d_y' % d for d in d_arr]
    methods += ['bpr_%d' % d for d in d_arr]
    methods += ['crowdbt', 'bt']

    ndcg5_data = []
    ndcg10_data = []

    for data_cat in data_cats:
        for data_key in data_keys[data_cat]:
            if data_key == 'olympic' and (data_dir_name == 'data_20000' or data_dir_name == 'data_e100'):
                continue

            truth_df = pd.read_csv(Path('../data/') / data_cat / data_key / 'truth.tsv', index_col=0, sep='\t')

            if data_dir_name == 'data':
                result_dir_sub = result_dir / data_cat / data_key
            else:
                result_dir_sub = result_dir / data_cat / data_key / str(trial)

            for method in methods:
                print(data_key, method)
                if method == 'bt' or method == 'crowdbt':
                    for param in [0.001, 0.01, 0.1]:
                        if method == 'bt':
                            p = np.loadtxt(result_dir_sub / 'bt_x_lambda{}.dat'.format(param))
                        elif method == 'crowdbt':
                            p = np.loadtxt(result_dir_sub / 'crowdbt_x_lambda{}.dat'.format(param))

                        for viewpoint in truth_df.columns:
                            if viewpoint[0] != '*':
                                continue

                            p_truth = truth_df.loc[:, viewpoint].values
                            rm = RankingMeasures(p, p_truth)
                            ndcg5 = rm.nDCG(5)
                            ndcg10 = rm.nDCG(10)

                            ndcg5_data.append({'data_key': data_key, 'ndcg5': ndcg5,
                                               'method': method, 'viewpoint': viewpoint,
                                               'param': param})
                            ndcg10_data.append({'data_key': data_key, 'ndcg10': ndcg10,
                                                'method': method, 'viewpoint': viewpoint,
                                                'param': param})

                elif 'bpr' in method:
                    d = int(method.split('_')[-1])
                    for param in [0.001, 0.01, 0.1]:
                        x = np.loadtxt(result_dir_sub / 'bpr_x_lambda{}_d{}.dat'.format(param, d))

                        for viewpoint in truth_df.columns:
                            if viewpoint[0] != '*':
                                continue

                            p_truth = truth_df.loc[:, viewpoint].values
                            ndcg5, ndcg10 = get_best_ndcg(x, p_truth)

                            ndcg5_data.append({'data_key': data_key, 'ndcg5': ndcg5,
                                               'method': method, 'viewpoint': viewpoint,
                                               'param': param})
                            ndcg10_data.append({'data_key': data_key, 'ndcg10': ndcg10,
                                               'method': method, 'viewpoint': viewpoint,
                                               'param': param})

                elif 'blade_chest' in method:
                    d = int(method.split('_')[-2])
                    m = method[-1]
                    for param in [0.001, 0.01, 0.1]:
                        x = np.loadtxt(result_dir_sub / 'blade_chest_{}_lambda{}_d{}.dat'.format(m, param, d))

                        for viewpoint in truth_df.columns:
                            if viewpoint[0] != '*':
                                continue

                            p_truth = truth_df.loc[:, viewpoint].values
                            ndcg5, ndcg10 = get_best_ndcg(x, p_truth)

                            ndcg5_data.append({'data_key': data_key, 'ndcg5': ndcg5,
                                               'method': method, 'viewpoint': viewpoint,
                                               'param': param})
                            ndcg10_data.append({'data_key': data_key, 'ndcg10': ndcg10,
                                               'method': method, 'viewpoint': viewpoint,
                                               'param': param})

                else:
                    d = int(method.split('_')[-1])
                    x = np.loadtxt(result_dir_sub / 'crowdea_x_alpha0.1_d{}.dat'.format(d))

                    for viewpoint in truth_df.columns:
                        if viewpoint[0] != '*':
                            continue

                        p_truth = truth_df.loc[:, viewpoint].values
                        ndcg5, ndcg10 = get_best_ndcg(x, p_truth)

                        ndcg5_data.append({'data_key': data_key, 'ndcg5': ndcg5,
                                           'method': method, 'viewpoint': viewpoint})
                        ndcg10_data.append({'data_key': data_key, 'ndcg10': ndcg10,
                                           'method': method, 'viewpoint': viewpoint})

    Path('../ndcg/').mkdir(parents=True, exist_ok=True)
    if data_dir_name == 'data':
        pd.DataFrame(ndcg5_data).to_csv('../ndcg/ndcg5_%s.csv' % (data_dir_name))
        pd.DataFrame(ndcg10_data).to_csv('../ndcg/ndcg10_%s.csv' % (data_dir_name))
    else:
        pd.DataFrame(ndcg5_data).to_csv('../ndcg/ndcg5_%s_%d.csv' % (data_dir_name, trial))
        pd.DataFrame(ndcg10_data).to_csv('../ndcg/ndcg10_%s_%d.csv' % (data_dir_name, trial))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir_name', type=str)
    parser.add_argument('trial', type=int)
    args = parser.parse_args()

    main(args.data_dir_name, args.trial)
