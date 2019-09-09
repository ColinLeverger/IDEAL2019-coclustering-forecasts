import logging
import os

import numpy as np
import pandas as pd

from clustering_machine import ClusteringMachine
from file_helper import FileHelper
from hmm_machine import HmmMachine
from predict_machine import PredictMachine
from my_utils import MyUtils

pd.options.mode.chained_assignment = None


class ECMLMachine:

    def __init__(self, file_name):
        self.utils_cl = MyUtils()
        self.this_file_dir = os.path.dirname(os.path.realpath(__file__))
        self.fh = FileHelper()
        self.hm = HmmMachine()
        self.cm = ClusteringMachine()
        self.pm = PredictMachine()
        self.my_metric = "euclidean"
        self.file_name = file_name
        self.out_fcasts_f = os.path.join(self.this_file_dir, "res", "fcasts", file_name, "ecml")
        self.fh.ensure_dirs_exist([self.out_fcasts_f])
        logging.info("Instantiated ECML_operator")

    def get_results_univ(self, df, mean_day, n_pt_one_period, n_serie_concatenated = 1):
        # Create datasets in the format we need
        logging.info("Computing ECML results")
        logging.info("Find best number of cluster")
        (df_train, df_valid, df_test)     = self.utils_cl.app_valid_test(df, n_pt_one_period)
        (df_train_compar, df_test_compar) = self.utils_cl.app_test(df, n_pt_one_period)
        
        last_day = df_train_compar["val_"][-len(df_test_compar):]

        # run algos using df_valid to find the better num of kmeans tb-used
        mean_mse = self.do_it(
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 70, 80, 100, 150, 200],
            df_train, df_valid, mean_day,
            n_pt_one_period)

        best_num_of_kmean = min(mean_mse, key = lambda t: t[1])[0]
        logging.info("Found best number of cluster: %s", best_num_of_kmean)
        # run algos using df_test with previously found best kmean
        res = self.do_it(
            [best_num_of_kmean], 
            df_train, df_test, mean_day,
            n_pt_one_period)

        # Retrieve results
        (mse_colin, mse_fake, mse_mean) =    (res[0][1], res[0][2], res[0][3])
        (mae_colin, mae_fake, mae_mean) =    (res[0][4], res[0][5], res[0][6])
        (mase_colin, mase_fake, mase_mean) = (res[0][7], res[0][8], res[0][9])

        (std_mse_colin, std_mse_fake, std_mse_mean) =    (res[0][10], res[0][11], res[0][12])
        (std_mae_colin, std_mae_fake, std_mae_mean) =    (res[0][13], res[0][14], res[0][15])
        (std_mase_colin, std_mase_fake, std_mase_mean) = (res[0][16], res[0][17], res[0][18])
        logging.info("Retrieve results")

        # Compute baselines
        logging.info("Compute baselines")
        (_, mse_ar_error, mae_ar_error, mase_ar_error)  = self.pm.do_forecast_ar_model(
            last_day, df_train_compar["val_"], df_test_compar["val_"])
        (mse_hw, mae_hw, mase_hw)  = -1, -1, -1

        return (
            mse_colin, mse_fake, mse_mean, 
            mae_colin, mae_fake, mae_mean,
            mase_colin, mase_fake, mase_mean,
            mse_ar_error, mae_ar_error, mase_ar_error, 
            mse_hw, mae_hw, mase_hw,
            best_num_of_kmean,
            std_mse_colin, std_mse_fake, std_mse_mean,
            std_mae_colin, std_mae_fake, std_mae_mean, 
            std_mase_colin, std_mase_fake, std_mase_mean
        )
        
    def do_it(self, 
              km_sizes, 
              df_train, df_valid_ou_test, mean_day,
              n_pt_one_period, n_serie_concatenated = 1):
        mean_mse = []
        for my_km_size in km_sizes:
            logging.info("ECML with km_size of %s", my_km_size)
            n_day_found_train = len(df_train["n_day_"].unique())
            logging.debug("There are %s days in train", n_day_found_train)

            #################################
            # I. TRAIN 
            #################################
            (km, comprehensive_clusters_df_train) = self.cm.do_kmeans_wrapper(
                df_train,
                km_size = my_km_size
            )

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # MM
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # compute MMs
            raw_hmm_data_df_train = list(
                self.hm.compute_raw_hmm(comprehensive_clusters_df_train, order = 1))
            # compute transition matrix
            transition_mat = self.hm.compute_hmm_transition_mat_1d(
                raw_hmm_data_df_train, my_km_size)

            #################################
            # II. VALID
            #################################
            # 1. apply km on df_valid data
            y_pred = self.cm.apply_clustering(df_valid_ou_test, km)

            # create tuple of known/wanted:
            # a) for ts data itself
            s_arr=[]
            for c in range(df_valid_ou_test["n_day_"].min() + 1, df_valid_ou_test["n_day_"].max() - 1):
                s_arr.append(
                    (
                        df_valid_ou_test[df_valid_ou_test["n_day_"] == c - 1]["val_"].values, 
                        df_valid_ou_test[df_valid_ou_test["n_day_"] == c]["val_"].values,
                        df_valid_ou_test[df_valid_ou_test["n_day_"] <= c]["val_"].values
                    )
                )
            
            # b) for ts labels
            s_l_arr=[]
            for c in range(1, len(y_pred) - 1):
                s_l_arr.append((y_pred[c - 1], y_pred[c]))

            precision_colin = [] 
            precision_fake  = []
            precision_mean  = []

            precision_colin_mae = [] 
            precision_fake_mae  = []
            precision_mean_mae  = []

            precision_colin_mase = [] 
            precision_fake_mase  = []
            precision_mean_mase  = []

            # compute predictions and mse
            for count in range(0, len(s_arr)):
                path_count = os.path.join(self.out_fcasts_f, str(count) + "_fcast_" + str(my_km_size) + "_kmsize.csv") 
                (known, guess, before_w)     = s_arr[count]
                known_w = np.pad(known, (0,len(before_w) - len(known)), "constant", constant_values=-42.42)
                guess_w = np.pad(guess, (0,len(before_w) - len(guess)), "constant", constant_values=-42.42)
                (known_l, guess_l) = s_l_arr[count]                            

                pd.DataFrame({"known": known_w[:], "guess": guess_w[:], "before": before_w[:]}) .to_csv(path_count,  sep = ";", index_label = False, index = False)

                pred = self.pm.predict_median_hmm(known_l, transition_mat, km)
                pred_fake = self.pm.predict_median_hmm(
                    known_l, transition_mat, km, 
                    real_class_of_following_day = guess_l) 
                
                precision_colin.append(self.utils_cl.compute_mse(guess, pred))
                precision_fake .append(self.utils_cl.compute_mse(guess, pred_fake))
                precision_mean .append(self.utils_cl.compute_mse(guess, mean_day))

                precision_colin_mae.append(self.utils_cl.compute_mae(guess, pred))
                precision_fake_mae .append(self.utils_cl.compute_mae(guess, pred_fake))
                precision_mean_mae .append(self.utils_cl.compute_mae(guess, mean_day))

                precision_colin_mase.append(self.utils_cl.compute_mase(known, guess, pred))
                precision_fake_mase .append(self.utils_cl.compute_mase(known, guess, pred_fake))
                precision_mean_mase .append(self.utils_cl.compute_mase(known, guess, mean_day))                        

            mean_mse.append(
                (
                    my_km_size, np.mean(precision_colin), np.mean(precision_fake), np.mean(precision_mean),
                    np.mean(precision_colin_mae), np.mean(precision_fake_mae), np.mean(precision_mean_mae),
                    np.mean(precision_colin_mase), np.mean(precision_fake_mase),np.mean(precision_mean_mase),
                    np.std(precision_colin), np.std(precision_fake), np.std(precision_mean),
                    np.std(precision_colin_mae), np.std(precision_fake_mae), np.std(precision_mean_mae),
                    np.std(precision_colin_mase), np.std(precision_fake_mase), np.std(precision_mean_mase)
                )
            )
        return mean_mse
