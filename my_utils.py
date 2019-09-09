# utils
import datetime
import logging
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
np.seterr(divide='ignore', invalid='ignore')


class MyUtils:

    @staticmethod
    def init():
        logging.info("Utils instance")

    @staticmethod
    def rd(n, decimals=5):
        """floor with numeric precision for results"""
        multiplier = 10 ** decimals
        return math.floor(n * multiplier) / multiplier

    @staticmethod
    def find_best_perf_by_index(res_arr, ncf):
        """
        Extract from a complex array of results the number of cluster linked to
        the best performances in terms of MSE.

        NOTE (why do we use min): sometimes, MODL founds less clusters when  it retrains coclustering (e.g w/ train_and_val altogether). We need to be sure we do not go to an infinite loop if it happens : if the best num of cluster found for this data is the highest of the train, thus could be higher than the highest of the train_and_val, we need to deal with that. 
        """
        best_perf = 0
        best_num_clust = 0
        a = []
        for e in res_arr:
            if e[3][0][1] > best_perf:
                a = e
                best_num_clust = e[2]
                best_perf = e[3][0][1]
        return a, min(best_num_clust, ncf)

    def compute_mse(self, a, b):
        """
        Compute mean square error between two given vectors.
        """
        return self.rd(mean_squared_error(a, b))
    
    def compute_mae(self, a, b):
        """
        Compute mean absolute error between two given vectors.
        """
        return self.rd(mean_absolute_error(a, b))

    def compute_mase(self, y_true_yester, y_true, y_pred): 
        """
        Compute mean absolute scaled error between two given vectors.
        """
        mae = self.compute_mae(y_true, y_pred)
        mae_naive = self.compute_mae(y_true_yester, y_true)
        return self.rd(mae / mae_naive)

    @staticmethod
    def nans_to_mean(a):
        """
        Fill nas of a given dataframe with mean values.
        """
        return a.fillna(a.mean())
   
    @staticmethod
    def drop_nans(a):
        """
        Drop nas of a given dataframe.
        """
        return a.dropna()

    @staticmethod
    def my_sort_func(array_of_tupple, key_ind):
        """
        Sort an array of tuple knowing which tuple's element to use for sort
        """
        return sorted(array_of_tupple, key=lambda tup: tup[key_ind])

    @staticmethod
    def get_biggest_contiguous_group(splitted):
        """
        From a list of all the days present in a dataset, gather all groups of
        days that are contiguous. Return biggest group of days.
        """
        filtered = []
        subtable = []

        for i in range(len(splitted) - 1):
            f = splitted[i]
            s = splitted[i + 1]

            if f["n_day_"].iloc[0] + 1 == s["n_day_"].iloc[0]:
                subtable.append(f)
            else:
                filtered.append(subtable)
                subtable = []
        filtered.append(subtable)
        return max(filtered, key=len)

    @staticmethod
    def find_group(cluster_and_values, num_day):
        """
        Find the group of given num_day.

        PARAMETERS
        ----------------
        - cluster_and_values: dic(cluster_num => values_associated)
        - num_day: int

        DIRTY TURNAROUND:
        ----------------
        There is a mess with num_day type and I did not have time to fix the
        mess so I have hacked the mess with the three lines of this function...
        """
        l = [g for g, v in cluster_and_values.items() if num_day in v]
        if not l:
            l = [g for g, v in cluster_and_values.items() if str(num_day) in v]
        # get the group of -1 if there is not
        return next(iter(l), "not_predicted")

    @staticmethod
    def find_first_and_last_date_of_ts(ts, date_name ="date_"):
        """ 
        Find the day after first day and the day before last days' date of a
        given time series. 
        Dont return very first and very last day (because of
        incomplete data)
        """
        first = pd.to_datetime(
            ts[date_name].iloc[0]) + datetime.timedelta(days=1)
        last =  pd.to_datetime(
            ts[date_name].iloc[-1]) + datetime.timedelta(days=-1)
        return first, last

    def rm_f_and_l_day(self, df, date_name = "date_"):
        """
        Get rid of very first and very last day of dataframe (because of
        incomplete data)
        """
        (f, l) = self.find_first_and_last_date_of_ts(df)
        df[date_name] = pd.to_datetime(df[date_name])
        mask = (df[date_name] > f) & (df[date_name] <= l)
        return df.loc[mask]

    @staticmethod
    def app_valid_test(df, len_one_day):
        """
        From a given df, creates and returns a 70 15 15 ensemble for 
        train, test and validation.
        """
        df = df.sort_values("n_day_")
        n_days = len(df)        

        # do the cuts
        sev = int(n_days * 0.7) - 1
        sev = sev - (sev % len_one_day)
        hui = int(n_days * 0.85) - 1
        hui = hui - (hui % len_one_day)

        return df[0:sev], df[sev:hui], df[hui:]

    @staticmethod
    def app_test(df, len_one_day):
        """
        From a given df, creates and returns a 85 15 ensemble for 
        train and test (no valid).
        """
        df = df.sort_values("n_day_")
        n_days = len(df)

        # do the cuts
        sev = int(n_days * 0.85) - 1
        sev = sev - (sev % len_one_day)

        return df[0:sev], df[sev:]

    @staticmethod
    def remove_elements_from_array(arr, elems):
        return [elem for elem in arr if elem not in elems]

    def flattify(self, l, order = 1):
        flat = [item for sublist in l for item in sublist]
        if order == 1:
            return flat
        else:
            return self.flattify(flat, order - 1)

    """LOADED PANDAS DATASET MANIPULATION"""
    def format_data_for_classifier(self, df, cluster_and_values, label):
        """
        This method takes as input all df loaded and cluster_and_value object
        to process it in order to create a training ensemble for classification.

        PARAMETERS FROM EXAMPLE
        ------------------------
          * df: entire df with all days
          * cluster_and_values: dic(cluster_num => values_associated)
        """
        logging.info("Formating data for classifier w/ dataset %s...", label)
        # Get all days ids in this day
        # FIXME 5000 is empirical?
        inds = list(range(1, 5000))

        # Considere that all days will be the same len as the first day in
        # dataset
        first_day_id = df.head(1)["n_day_"].values[0]
        len_day_ref = len(df[df["n_day_"] == first_day_id])

        cols = list(range(0, len_day_ref)) + ["y"]
        # init a res dataframe with known cols and ind
        res = pd.DataFrame(
            columns=cols,
            index=inds)

        # For each day, enrich data and go from column based data to
        # line based data in other words, create dataset for bayesian
        # learning
        for (_, day) in df.groupby(df["date_"]):
            if len(day) == len_day_ref:
                num_day = int(day.iloc[0]["n_day_"])
                num_day_tom = num_day + 1
                group_tom = self.find_group(cluster_and_values, num_day_tom)

                if group_tom is not "not_predicted":
                    res.loc[num_day] = [format(float(i), '.1f') for i in list(day["val_"].values)] + [group_tom]
            else:
                logging.info("This day is too small: %s != %s", len(day), len_day_ref)

        return res.dropna()
