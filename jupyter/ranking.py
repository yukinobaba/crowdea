#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections
import scipy.stats

class RankingMeasures:
    def __init__(self, hypos, refs):
        assert len(hypos) == len(refs)
        self.hypos = np.array(hypos)
        self.refs = np.array(refs)
        self.hypo_ranking = self.get_ranking(hypos)
        self.ref_ranking = self.get_ranking(refs)

    def get_ranking(self, values):
        # ranking = (rank of i_th item)
        idxs = np.argsort(values)[::-1]
        ranking = np.zeros(len(idxs))
        for i, idx in enumerate(idxs): ranking[idx] = i
        ties = collections.defaultdict(list)
        for idx in idxs: ties[values[idx]].append(idx)
        for tie in ties.values():
            s = np.mean([ranking[idx] for idx in tie])
            for idx in tie: ranking[idx] = s
        return ranking

    def nDCG(self, k):
        idxs = np.argsort(self.hypos)[::-1][:k]
        refs_k = self.refs[idxs]
        return self.DCG(refs_k) / self.DCG(sorted(self.refs, reverse=True)[:k])

    def DCG(self, values):
        n = len(values)
        return np.sum([values[i-1] / np.log2(i+1) for i in range(1, n+1)])

if __name__ == '__main__':
    hypos = [6, 5, 4, 3, 2, 1]
    refs = [3, 2, 3, 0, 1, 2]
    rm = RankingMeasures(hypos, refs)
    print(rm.nDCG(1))
