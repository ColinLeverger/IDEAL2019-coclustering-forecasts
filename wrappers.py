#!/usr/bin/env python3
import datetime as df_train
# Import for logs
import logging
import math

# Create now variable w/ info about time of exec
n = df_train.datetime.now()
tstmp = (str(n.year) + str(n.month) + str(n.day) + str(n.hour) + str(n.minute))
logging.info("Start Application. Prefix for res file is %s", tstmp)

import pandas as pd

from my_utils import MyUtils

from clustering_machine import ClusteringMachine


class Wrappers:

    def __init__(self, km, fh):
        self.utils_cl = MyUtils()
        self.km = km
        self.fh = fh
        self.cm = ClusteringMachine()

        logging.info("Wrappers instantiated")

    def launch_preds(self,
            data_fmt_clustering_train, data_fmt_clustering_valid_or_test, # Train and test formated to be fed in classifier
            days_train, days_valid_or_test, l_ref, # Days of train and test in a list
            cluster_and_values_train, # To compute centroids
            nb_cluster, mean_day,
            fit, classifier):
        """
        Utilitary piece of code that is used to launch predictions. This wrapper
        takes input from clustering or coclustering algorithms and generates
        associated predictions MSEs.

        This method works both for clustering and coclustering group creation.

        ARGS:
        * data_fmt_clustering_train: Formatted data, ready to be fed into a
        classifier
        * data_fmt_clustering_valid_or_test: Same, but with only the last 20% remaining
        * days_train: all days present in train
        * days_valid_or_test: all days present in test
        * cluster_and_values_train: dic(cluster_num => values_associated)
        * nb_cluster: how many clusters pour cette passe
        * mean_day: mean day on train only
        * fit: classifier used
        * classifier: string label, which classifier are we using?
        """
        # Separate target from data
        train_full   = data_fmt_clustering_train.loc[:, data_fmt_clustering_train.columns != 'y'].sort_index()
        train_target = data_fmt_clustering_train.ix[:, "y"].sort_index()

        test_full   = data_fmt_clustering_valid_or_test.loc[:, data_fmt_clustering_valid_or_test.columns != 'y'].sort_index()
        test_target = data_fmt_clustering_valid_or_test.ix[:, "y"].sort_index()
        
        # Feed df_train
        fit.fit(train_full, train_target)

        res = {}

        # Retrieve cluster names (we are not numbering them but referencing them
        # their little given name)
        clusts_names = cluster_and_values_train.keys()

        (centroids, e) = self.km.compute_centroids(cluster_and_values_train, days_train, l_ref)

        # NOTE: this lines below are indeed strange... But we have had developed some other API that we had to follow.
        # That is why we format the data that way.
        y_pred = pd.DataFrame(fit.predict_proba(test_full), columns=fit.classes_)
        y_pred["Predictedy"] = y_pred.idxmax(axis=1)
        y_pred.columns = ["Proby" + str(col) for col in y_pred.columns]
        y_pred.rename(columns={'ProbyPredictedy':'Predictedy'}, inplace=True)

        # 1. w/ MODL & probabilistic prevision                 
        res[False] = self.km.process_pred_proba(
            y_pred, test_target, clusts_names,
            centroids, days_valid_or_test,  
            nb_cluster, l_ref,
            mean_day)
        
        # 2. Oracle
        y_pred_or = y_pred.copy().reset_index(drop = True)
        y_pred_or["Predictedy"] = test_target.reset_index(drop = True)

        res[True] = self.km.process_pred_proba(
            y_pred_or, test_target, clusts_names,
            centroids, days_valid_or_test,   
            nb_cluster, l_ref,
            mean_day, o = True)

        return (
            e, 
            res
        )

    def simplify_coclus(self, 
        n_clus_found, n_clus_target, mpi, mcn,
        file_name_train, path_file_test, file_name_root, wlabel):
        """
        Ad-hoc method to simplify a Khiops coclustering json file.
        """
        # loop configuration
        to_add = 5
        to_remove = 2 
        mcn_init_ok = False

        # First try to get the good number of cluster!
        logging.info("Keep %s percent of info", mpi)
        logging.info("Desired nb of cluster is %s", n_clus_target)

        identifier = mpi
        
        # While we do not reach the good number of cluster
        all_mpis = []
        while n_clus_found != n_clus_target:
            if mcn == 1 and len(all_mpis) < 20:
                logging.info("Simplifying using MPI")
                simplified_file = self.km.simplify_coclustering(file_name_train, mpi = mpi)
                n_clus_found = self.km.get_cluster_number(file_name_train, ref_id = mpi)
                logging.info("Found %s clusters in simplified coclus file (expected %s)", n_clus_found, n_clus_target)
                all_mpis.append(mpi)
                if n_clus_found > n_clus_target:
                    self.fh.rm_simplified_outputs(file_name_train, mpi)
                    mpi = mpi - to_remove
                    logging.info("MPI was apparently too big, decreasing its size of -%s: mpi = %s", to_remove, mpi)  
                    if mpi in all_mpis:
                        to_remove = to_remove / 2
                        to_add = to_add / 2    
                elif n_clus_found < n_clus_target:
                    self.fh.rm_simplified_outputs(file_name_train, mpi)
                    mpi = mpi + to_add
                    logging.info("MPI was apparently too small, increasing its size of +%s: mpi = %s", to_add, mpi) 
                identifier = mpi
            else:
                logging.info("Now simplifying using MCN")

                # Init new loop: should use cell number now because it is more
                # precise than the latest
                if not mcn_init_ok:
                    self.km.simplify_coclustering(file_name_train, mpi = mpi)
                    mcn = self.km.get_cells_number(file_name_train, mpi)
                    to_add = math.floor(mcn * 0.2437)  # empirical coefficient
                    to_remove = math.floor(mcn * 0.097)
                    mcn = mcn + to_add
                    # once initialised, we do not want to re run above
                    mcn_init_ok = True
                simplified_file = self.km.simplify_coclustering(file_name_train, mcn = mcn)
                n_clus_found = self.km.get_cluster_number(file_name_train, ref_id = mcn)
                logging.info("Found %s clusters in simplified coclus file (expected %s)", n_clus_found, n_clus_target)
                if n_clus_found > n_clus_target:
                    self.fh.rm_simplified_outputs(file_name_train, mcn)
                    mcn = mcn - to_remove
                    logging.info("MCN was apparently too big, decreasing its size of -%s: mcn = %s", to_remove, mcn)  
                elif n_clus_found < n_clus_target:
                    self.fh.rm_simplified_outputs(file_name_train, mcn)
                    mcn = mcn + to_add
                    logging.info("MCN was apparently too small, increasing its size of +%s: mcn = %s", to_add, mcn) 
                identifier = mcn

        
        # Then, manage khiops files
        _, cluster_and_values_train = self.km.get_clusters(simplified_file)
        # Write labels of test file for future MODL transfert
        path_test_labels = self.fh.write_labels(path_file_test, identifier)   
        # Write cluster and values on disk to compare clustering results    
        self.fh.write_cav_on_disk(cluster_and_values_train, file_name_root, n_clus_found, wlabel)     
        
        # Deploy dic and use deployed dic to transfer (transfer = actually use
        # MODL model created, fm)
        path_deployed_dic = self.km.deploy_coclustering(file_name_train, identifier)
        path_transfered   = self.km.transfer_database(path_deployed_dic, path_test_labels, path_file_test, identifier) 

        # Retrieve clusters attribution from khiops use of MODL
        cluster_and_values_test = self.km.get_clusters_from_dep(path_transfered)

        return (
            (mpi, mcn, n_clus_found),
            (cluster_and_values_train, cluster_and_values_test),
            simplified_file
        )

    def modl_wrap(self, 
        df_train, df_valid_or_test, 
        l_ref, mean_day,
        fit, classifier,
        cluster_and_values_train, cluster_and_values_valid_or_test, n_clus_found):

        # Create datasets before training a classifier
        data_fmt_train = self.utils_cl.format_data_for_classifier(df_train, cluster_and_values_train, "days_train") 
        data_fmt_valid_or_test  = self.utils_cl.format_data_for_classifier(df_valid_or_test, cluster_and_values_valid_or_test, "days_valid_or_test")

        (_, res) = self.launch_preds(
                data_fmt_train, data_fmt_valid_or_test,
                df_train, df_valid_or_test, l_ref,
                cluster_and_values_train,
                n_clus_found, mean_day,
                fit, classifier)

        # Extract results
        # For the simple algo (no oracle)  
        (mses_non_proba, mses_proba, mean_mse_mean_day)  = res[False][0]        
        (mae_non_proba,  mae_proba,  mean_mae_mean_day)  = res[False][1]
        (mase_non_proba, mase_proba, mean_mase_mean_day) = res[False][2]
        classifier_acc = res[False][3]
        (std_mse_non_proba, std_mse_proba, std_mse_mean_day)  = res[False][4]        
        (std_mae_non_proba, std_mae_proba,  std_mae_mean_day)  = res[False][5]
        (std_mase_non_proba, std_mase_proba, std_mase_mean_day)  = res[False][6]

        # Then for the oracle algo
        (mse_non_proba_or,  _, _) = res[True][0]
        (mae_non_proba_or,  _, _) = res[True][1]
        (mase_non_proba_or, _, _) = res[True][2]
        (std_mse_non_proba_or, _, _)  = res[True][4]        
        (std_mae_non_proba_or, _,  _)  = res[True][5]
        (std_mase_non_proba_or, _, _)  = res[True][6]

        return (
            (                    
                (mses_non_proba, mses_proba, mae_non_proba, mae_proba, mase_non_proba, mase_proba),
                (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or),
                (mean_mse_mean_day, mean_mae_mean_day, mean_mase_mean_day),
                classifier_acc,                    
                (std_mse_non_proba, std_mse_proba, std_mse_mean_day, std_mae_non_proba, std_mae_proba, std_mae_mean_day, std_mase_non_proba, std_mase_proba, std_mase_mean_day),
                (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or)
            )
        )

    def clustering_wrapper(self,
        func_and_label, 
        days_train, days_valid_or_test, mean_day_train,
        l_ref, file_name, nb_cluster,
        fit, classifier):
        func, wlabel = func_and_label
        logging.info("Clustering wrapper: process %s redictions for file %s with %s classifier", wlabel, file_name, classifier)

        # TRAIN: train clustering model
        (clustering, clusters_train) = func(days_train, nb_cluster) 
        cc_train                 = self.cm.create_c_a_v(days_train, clusters_train) 

        # TEST: apply the clustering model previously trained
        clusters_test = self.cm.apply_clustering(days_valid_or_test, clustering)    
        cc_valid_or_test = self.cm.create_c_a_v(days_valid_or_test, clusters_test) 

        cav_clustering_train, _ = self.cm.format_clust_results(nb_cluster, cc_train)
        self.fh.write_cav_on_disk(cav_clustering_train, file_name, nb_cluster, wlabel)
        cav_clustering_valid_or_test , _ = self.cm.format_clust_results(nb_cluster, cc_valid_or_test)
                        
        data_fmt_clustering_train = self.utils_cl.format_data_for_classifier(days_train, cav_clustering_train, "days_train") 
        data_fmt_clustering_valid_or_test  = self.utils_cl.format_data_for_classifier(days_valid_or_test,  cav_clustering_valid_or_test, "days_valid_or_test") 
        
        (empty_cluster_found, res) = self.launch_preds(
            data_fmt_clustering_train, data_fmt_clustering_valid_or_test,
            days_train, days_valid_or_test, l_ref, 
            cav_clustering_train, 
            nb_cluster, mean_day_train, 
            fit, classifier)

        # Then for the simple algo (no oracle)  
        (mses_non_proba, mses_proba, mean_mse_mean_day)  = res[False][0]        
        (mae_non_proba,  mae_proba,  mean_mae_mean_day)  = res[False][1]
        (mase_non_proba, mase_proba, mean_mase_mean_day) = res[False][2]
        classifier_acc = res[False][3]
        (std_mse_non_proba, std_mse_proba, std_mse_mean_day)  = res[False][4]        
        (std_mae_non_proba, std_mae_proba,  std_mae_mean_day)  = res[False][5]
        (std_mase_non_proba, std_mase_proba, std_mase_mean_day)  = res[False][6]            

        # Then for the oracle algo
        (mse_non_proba_or,  _, _) = res[True][0]
        (mae_non_proba_or,  _, _) = res[True][1]
        (mase_non_proba_or, _, _) = res[True][2]
        (std_mse_non_proba_or, _, _)  = res[True][4]        
        (std_mae_non_proba_or, _,  _)  = res[True][5]
        (std_mase_non_proba_or, _, _)  = res[True][6]

        return (     
            empty_cluster_found,
            (
                (mses_non_proba, mses_proba, mae_non_proba, mae_proba, mase_non_proba, mase_proba),
                (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or),
                (mean_mse_mean_day, mean_mae_mean_day, mean_mase_mean_day),
                classifier_acc,
                (std_mse_non_proba, std_mse_proba, std_mse_mean_day, std_mae_non_proba, std_mae_proba, std_mae_mean_day, std_mase_non_proba, std_mase_proba, std_mase_mean_day),
                (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or)
            )
        )
