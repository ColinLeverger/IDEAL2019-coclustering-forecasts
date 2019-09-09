# Import lib
import math
import logging

from my_utils import MyUtils

import numpy as np
import pandas as pd


class HmmMachine:

    def __init__(self):
        self.utils_cl = MyUtils()
        logging.info("HMM machine instantiated")

    def compute_raw_hmm(self, tuple_array, order=1):
        """
        From an array of typle ((srv1, 2), (srv2, 5), (srv3, 4), ...):
        * sort it,
        * create a List(2,5,4,...), 
        * compute HMM (2,5),(5,4), ...
        """
        # flatmap + only keep first elem of tuple; need to sort to exhibits the sequences of days!
        iterable = [x[1] for x in self.utils_cl.my_sort_func(tuple_array, 0)]

        i = iter(iterable)
        win = []
        for e in range(0, order + 1):
            win.append(next(i))
        yield win
        for e in i:
            win = win[1:] + [e]
            yield win

        return list(win)

    @staticmethod
    def compute_hmm_transition_mat_1d(raw_hmm_data, n_cluster):
        """
        Knowing the number of clusters in the KNN and with numeric results of
        the KNN, compute the HMM transition matrice, e.g. percentages of going
        from one state to another.
        """
        order = len(raw_hmm_data[0]) - 1

        pow2 = int(n_cluster * n_cluster)
        pow3 = int(n_cluster * n_cluster * n_cluster)

        if order == 1:
            d = pd.DataFrame(0, index=np.arange(n_cluster),
                             columns=range(0, n_cluster))
            for tup in raw_hmm_data:
                d.at[tup[0], tup[1]] = d.at[tup[0], tup[1]] + 1
            # sum of lines
            sum_col = d.sum(1)
            for i in range(0, n_cluster):
                for j in range(0, n_cluster):
                    if d.at[i, j] != 0 and sum_col[i] != 0:
                        perc = 100 * (d.at[i, j] / sum_col[i])
                    else:
                        perc = 0
                    if not math.isnan(perc):
                        d.at[i, j] = perc
                    else:
                        d.at[i, j] = 0
        elif order == 2:
            d = pd.DataFrame(0, index=np.arange(
                pow2), columns=range(0, n_cluster))

            # First construct raw data
            for tup in raw_hmm_data:
                d.at[tup[0] * n_cluster + tup[1], tup[2]] = d.at[tup[0] * n_cluster + tup[1], tup[2]] + 1
            sum_col = d.sum(1)

            # Then percentagise it
            for i in range(0,  pow2):
                for j in range(0, n_cluster):
                    if d.at[i, j] != 0 and sum_col[i] != 0:
                        perc = 100 * (d.at[i, j] / sum_col[i])
                    else:
                        perc = 0
                    perc = 100 * (d.at[i, j] / sum_col[i])
                    if not math.isnan(perc):
                        d.at[i, j] = perc
                    else:
                        d.at[i, j] = 0
        elif order == 3:
            d = pd.DataFrame(0, index=np.arange(
                pow3), columns=range(0, n_cluster))

            for tup in raw_hmm_data:
                d.at[tup[0] * pow2 + tup[1] * n_cluster + tup[2], tup[3]
                     ] = d.at[tup[0] * pow2 + tup[1] * n_cluster + tup[2], tup[3]] + 1
            sum_col = d.sum(1)

            for i in range(0, pow3):
                for j in range(0, n_cluster):
                    if d.at[i, j] != 0 and sum_col[i] != 0:
                        perc = 100 * (d.at[i, j] / sum_col[i])
                    else:
                        perc = 0
                    perc = 100 * (d.at[i, j] / sum_col[i])
                    if not math.isnan(perc):
                        d.at[i, j] = perc
                    else:
                        d.at[i, j] = 0
        return d

    @staticmethod
    def compute_hmm_transition_mat_2d(raw_hmm_data1, raw_hmm_data2, n_cluster):
        """
        Knowing the number of clusters in the KNN and with numeric results of
        the KNN, compute the HMM transition matrice, e.g. percentages of going
        from one state to another. Two dimentional (e.g. using two sequences of
        two TS to use more info for predictions)
        """
        order = len(raw_hmm_data1[0]) - 1

        pow2 = int(n_cluster * n_cluster)
        pow3 = int(n_cluster * n_cluster * n_cluster)
        pow4 = int(n_cluster * n_cluster * n_cluster * n_cluster)

        raw_hmm_data = list(zip(raw_hmm_data1, raw_hmm_data2))

        if order == 1:
            d = pd.DataFrame(0, index=np.arange(pow2),
                             columns=range(0, pow2))
            for (tup1, tup2) in raw_hmm_data:
                d.at[tup1[0] * n_cluster + tup2[0], tup1[1] * n_cluster + tup2[1]
                     ] = d.at[tup1[0] * n_cluster + tup2[0], tup1[1] * n_cluster + tup2[1]] + 1
            # sum of lines
            sum_col = d.sum(1)
            for i in range(0, pow2):
                for j in range(0, pow2):
                    perc = 100 * (d.at[i, j] / sum_col[i])
                    if not math.isnan(perc):
                        d.at[i, j] = perc
                    else:
                        d.at[i, j] = 0
        return d
