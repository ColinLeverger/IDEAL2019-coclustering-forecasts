#!/usr/bin/env python3
import datetime as dt
import math
import os

# Import for logs
import logging.config
this_file_dir  = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(
    this_file_dir, "conf", 'logging.conf')
logging.config.fileConfig(config_path)

# Create now variable w/ info about time of exec
n = dt.datetime.now()
tstmp = (str(n.year) + str(n.month) + str(n.day) + str(n.hour) + str(n.minute))
logging.info("Start Application. Prefix for res file is %s", tstmp)

# File mgmt: write on disk results
logging.info("Initialize paths")
out_path_res   = os.path.join(this_file_dir, "res")
in_dir         = os.path.join(this_file_dir, "data", "csvs")
out_dir_res    = os.path.join(this_file_dir, "res")
out_dir_khiops = os.path.join(this_file_dir, "res", "khiops_res")
out_dir_fcasts = os.path.join(this_file_dir, "res", "fcasts")
glob_csv_res   = os.path.join(out_path_res, "all.csv")
vali_csv_res   = os.path.join(out_path_res, "valid.csv")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Bunch of cleanings for empty files
from file_helper import FileHelper
fh = FileHelper(tstmp)
fh.clean_zips_folder()
fh.clean_res_folder(out_dir_res)
fh.ensure_dirs_exist([out_path_res, in_dir, out_dir_res, out_dir_khiops, out_dir_fcasts])

# After cleaning, zip the code which is executed now
fh.zip_code()

# Init objects
from khiops import KhiopsManager
km    = KhiopsManager()

from my_utils import MyUtils
utils = MyUtils()

from ecml_machine import ECMLMachine

from clustering_machine import ClusteringMachine
cm    = ClusteringMachine()

from wrappers import Wrappers
wrapper = Wrappers(km, fh)

# Read configuration file: which input data to use?
import json
with open(os.path.join(this_file_dir, "conf", "conf_lite.json"), 'r') as f:
    confs = json.load(f)

# Init res files
print(
    "mse_ecml_colin",  "mse_ecml_or",  "mse_ecml_mean", 
    "mae_ecml_colin",  "mae_ecml_or",  "mae_ecml_mean",
    "mase_ecml_colin", "mase_ecml_or", "mase_ecml_mean",
    "std_mse_colin", "std_mse_fake", "std_mse_mean",
    "std_mae_colin", "std_mae_fake", "std_mae_mean", 
    "std_mase_colin", "std_mase_fake", "std_mase_mean",
    "mse_ar_error", "mae_ar_error", "mase_ar_error", 
    "best_num_of_kmean_ecml",

    "classifier_acc", 
    "best_num_clust", "nb_cluster_found_100", "fold_better",

    "mse_non_proba", "mae_non_proba", "mase_non_proba", 
    "std_mse_non_proba", "std_mae_non_proba", "std_mase_non_proba",

    "mse_proba", "mae_proba", "mase_proba",
    "std_mse_proba", "std_mae_proba", "std_mase_proba",

    "mse_non_proba_or", "mae_non_proba_or", "mase_non_proba_or",
    "std_mse_non_proba_or", "std_mae_non_proba_or", "std_mase_non_proba_or",

    "mse_mean_day", "mae_mean_day", "mase_mean_day",
    "std_mse_mean_day",   "std_mae_mean_day",  "std_mase_mean_day",       

    "clustering", "classifier", "file_name", "nb_days",
    file=open(glob_csv_res, "a"), sep=';')
logging.info("Initialised %s new result file", glob_csv_res)

print(
    "mse_ecml_colin",  "mse_ecml_or",  "mse_ecml_mean", 
    "mae_ecml_colin",  "mae_ecml_or",  "mae_ecml_mean",
    "mase_ecml_colin", "mase_ecml_or", "mase_ecml_mean",
    "std_mse_colin", "std_mse_fake", "std_mse_mean",
    "std_mae_colin", "std_mae_fake", "std_mae_mean", 
    "std_mase_colin", "std_mase_fake", "std_mase_mean",
    "mse_ar_error", "mae_ar_error", "mase_ar_error", 
    "best_num_of_kmean_ecml",

    "classifier_acc", 
    "best_num_clust", "nb_cluster_found_100", "fold_better",

    "mse_non_proba", "mae_non_proba", "mase_non_proba", 
    "std_mse_non_proba", "std_mae_non_proba", "std_mase_non_proba",

    "mse_proba", "mae_proba", "mase_proba",
    "std_mse_proba", "std_mae_proba", "std_mase_proba",

    "mse_non_proba_or", "mae_non_proba_or", "mase_non_proba_or",
    "std_mse_non_proba_or", "std_mae_non_proba_or", "std_mase_non_proba_or",

    "mse_mean_day", "mae_mean_day", "mase_mean_day",
    "std_mse_mean_day",   "std_mae_mean_day",  "std_mase_mean_day",       

    "clustering", "classifier", "file_name", "nb_days",
    file=open(vali_csv_res, "a"), sep=';')
logging.info("Initialised %s new result file", vali_csv_res)

#####################################
#           Begin scripts           #
#####################################
for k, v in confs.items():   
    out_dir_khiops_adapted = os.path.join(out_dir_khiops, k)

    input_files_absolute_path = fh.get_files_paths_in_folder(
        os.path.join(in_dir, k), filter_extention = ".csv")   

    # for each files in input folder, do clustering and write results on disk
    for f in input_files_absolute_path:
        logging.info("Processing %s file/dataset", f)
 
        # load dataframe in pandas
        f_df = pd.read_csv(f, sep =";")
        n_pt_per_day = len(f_df[f_df["n_day_"] == 1])

        # Split A V T
        # Get all uniq identifiers fro n_day_ to find lenght
        # NOTE: up to now I am filtering w/ ordered and I assume that "n_day_"
        # starts at 1, also there is no holes in the data.
        all_ids = f_df["n_day_"].unique()
        nb_days = len(all_ids)

        p_split_70 = math.floor(nb_days * 0.70)
        p_split_85 = math.floor(nb_days * 0.85)

        days_train            = f_df[f_df["n_day_"]  <  p_split_70]
        days_valid            = f_df[(f_df["n_day_"] >= p_split_70) & (f_df["n_day_"] < p_split_85)]
        days_train_and_valid  = f_df[f_df["n_day_"]  <  p_split_85]
        days_test             = f_df[f_df["n_day_"]  >= p_split_85]
        
        # Compute mean day once and forall for train and train+valid ensemble.
        mean_day_train = days_train.groupby('time_')['val_'].agg('mean').values
        l_ref = len(mean_day_train)
        logging.info('Len ref for file %s is %s', f, l_ref)

        mean_day_train_and_valid = days_train_and_valid.groupby('time_')['val_'].agg('mean').values

        # get filename to follow up between things
        file_name = fh.get_file_name(f)
        ecml  = ECMLMachine(file_name)

        path_train           = fh.write_days_on_disk(days_train, file_name, "_train")
        path_valid           = fh.write_days_on_disk(days_valid, file_name, "_valid")
        path_train_and_valid = fh.write_days_on_disk(days_train_and_valid, file_name, "_train_and_valid")
        path_test            = fh.write_days_on_disk(days_test,  file_name, "_test")

        file_name_train = fh.get_file_name(path_train)
        file_name_valid = fh.get_file_name(path_valid)
        file_name_test  = fh.get_file_name(path_test)
        file_name_train_and_valid = fh.get_file_name(path_train_and_valid)

        # Declare classifiers
        fit_tree = DecisionTreeClassifier(
            criterion = "gini", 
            random_state = 42,
            max_depth = 96, min_samples_leaf = 5)
        fit_lr = LogisticRegression(
            random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=10000)
        fit_bayes =  GaussianNB()

        try:
            (
                mse_ecml_colin, mse_ecml_fake, mse_ecml_mean,
                mae_ecml_colin, mae_ecml_fake, mae_ecml_mean,
                mase_ecml_colin, mase_ecml_fake, mase_ecml_mean,
                mse_ar_error, mae_ar_error, mase_ar_error,
                _, _, _,
                best_num_of_kmean_ecml,
                std_mse_colin, std_mse_fake, std_mse_mean,
                std_mae_colin, std_mae_fake, std_mae_mean,
                std_mase_colin, std_mase_fake, std_mase_mean
            ) = ecml.get_results_univ(f_df, mean_day_train, n_pt_per_day)

            ###################################################
            #              C O C L U S T E R I N G            #
            ###################################################
            km.train_coclustering(path_train)
            km.train_coclustering(path_train_and_valid)

            # Retrieve maximum information available stats to configure loops
            cells_number_100      = km.get_cells_number(file_name_train)
            nb_cluster_found_100  = km.get_cluster_number(file_name_train)
            nb_cluster_found_100_train_and_val  = km.get_cluster_number(file_name_train_and_valid)

            if nb_cluster_found_100:
                mpi = 1
                mcn = 1
                desired_nb_cluster = 2
                nb_cluster_found = 0
                all_results_coclus = []

                while desired_nb_cluster <= nb_cluster_found_100:
                    # Step 1: make coclustering file less precise/simplification
                    ((mpi, mcn, nb_cluster_found), (cavt, cavvt), _) = wrapper.simplify_coclus(
                        nb_cluster_found, desired_nb_cluster, mpi, mcn,
                        file_name_train, path_valid, file_name, "train_modl"
                    )

                    # Step 2: for each classifier, process using simplified coclustering file
                    for (fit, classifier) in [(fit_bayes, "bayes"), (fit_tree, "tree"), (fit_lr, "linear_reg")]:
                        v = wrapper.modl_wrap(
                            days_train, days_valid,
                            l_ref, mean_day_train,
                            fit, classifier,
                            cavt, cavvt, nb_cluster_found)
                        all_results_coclus.append((fit, classifier, nb_cluster_found, v))

                    # Increment compteurs pour la boucle
                    desired_nb_cluster = desired_nb_cluster + 1

                logging.debug("All res for crossval for file %s are %s", file_name, all_results_coclus)

                for clab in ["bayes", "tree", "linear_reg"]:
                    logging.info("Processing %s clab for coclustering results", clab)
                    res_val = [t for t in all_results_coclus if t[1].startswith(clab)]
                    classifier = res_val[0][1]
                    fit = res_val[0][0]

                    # Find best number of clusters for this classifier
                    v_classifier, best_num_clust = utils.find_best_perf_by_index(
                        res_val, nb_cluster_found_100_train_and_val)

                    # Store intermediate resuts for valid
                    (
                        mse_non_proba, mse_proba, mae_non_proba,
                        mae_proba, mase_non_proba, mase_proba) = v_classifier[3][0]
                    (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or) = v_classifier[3][1]
                    (mse_mean_day, mae_mean_day, mase_mean_day) = v_classifier[3][2]
                    classifier_acc = v_classifier[3][3]
                    (
                        std_mse_non_proba, std_mse_proba, std_mse_mean_day,
                        std_mae_non_proba, std_mae_proba, std_mae_mean_day,
                        std_mase_non_proba, std_mase_proba, std_mase_mean_day) = v_classifier[3][4]
                    (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or) = v_classifier[3][5]

                    logging.info("Number max cluster found for train is %s", nb_cluster_found_100)
                    logging.info("Number max cluster found for train_and_val is %s", nb_cluster_found_100_train_and_val)
                    logging.info("best_num_clust (for %s) found is %s", classifier, best_num_clust)

                    fold_better = nb_cluster_found_100 != best_num_clust

                    print(
                        mse_ecml_colin,  mse_ecml_fake,  mse_ecml_mean,
                        mae_ecml_colin,  mae_ecml_fake,  mae_ecml_mean,
                        mase_ecml_colin, mase_ecml_fake, mase_ecml_mean,
                        std_mse_colin, std_mse_fake, std_mse_mean,
                        std_mae_colin, std_mae_fake, std_mae_mean,
                        std_mase_colin, std_mase_fake, std_mase_mean,
                        mse_ar_error, mae_ar_error, mase_ar_error,
                        best_num_of_kmean_ecml,

                        classifier_acc,
                        best_num_clust, nb_cluster_found_100, fold_better,

                        mse_non_proba, mae_non_proba, mase_non_proba,
                        std_mse_non_proba, std_mae_non_proba, std_mase_non_proba,

                        mse_proba, mae_proba, mase_proba,
                        std_mse_proba, std_mae_proba, std_mase_proba,

                        mse_non_proba_or, mae_non_proba_or, mase_non_proba_or,
                        std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or,

                        mse_mean_day, mae_mean_day, mase_mean_day,
                        std_mse_mean_day,   std_mae_mean_day,  std_mase_mean_day,

                        "modl", classifier, file_name, nb_days,
                        file=open(vali_csv_res, "a"), sep=';')

                    # Process with train when best_num_clust is found
                    if best_num_clust != 0:
                        ((_, _, _), (cavt, cavvt), _) = wrapper.simplify_coclus(
                            0, best_num_clust, 1, 1,
                            file_name_train_and_valid, path_test, file_name, "train_val_modl_" + classifier
                        )
                        v_classifier = wrapper.modl_wrap(
                            days_train_and_valid, days_test,
                            l_ref, mean_day_train_and_valid,
                            fit, classifier,
                            cavt, cavvt, best_num_clust)

                        # Store results on disk
                        (
                            mse_non_proba, mse_proba, mae_non_proba,
                            mae_proba, mase_non_proba, mase_proba) = v_classifier[0]
                        (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or) = v_classifier[1]
                        (mse_mean_day, mae_mean_day, mase_mean_day) = v_classifier[2]
                        classifier_acc = v_classifier[3]
                        (
                            std_mse_non_proba, std_mse_proba, std_mse_mean_day,
                            std_mae_non_proba, std_mae_proba, std_mae_mean_day,
                            std_mase_non_proba, std_mase_proba, std_mase_mean_day) = v_classifier[4]
                        (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or) = v_classifier[5]

                        print(
                            mse_ecml_colin,  mse_ecml_fake,  mse_ecml_mean,
                            mae_ecml_colin,  mae_ecml_fake,  mae_ecml_mean,
                            mase_ecml_colin, mase_ecml_fake, mase_ecml_mean,
                            std_mse_colin, std_mse_fake, std_mse_mean,
                            std_mae_colin, std_mae_fake, std_mae_mean,
                            std_mase_colin, std_mase_fake, std_mase_mean,
                            mse_ar_error, mae_ar_error, mase_ar_error,
                            best_num_of_kmean_ecml,

                            classifier_acc,
                            best_num_clust, nb_cluster_found_100, fold_better,

                            mse_non_proba, mae_non_proba, mase_non_proba,
                            std_mse_non_proba, std_mae_non_proba, std_mase_non_proba,

                            mse_proba, mae_proba, mase_proba,
                            std_mse_proba, std_mae_proba, std_mase_proba,

                            mse_non_proba_or, mae_non_proba_or, mase_non_proba_or,
                            std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or,

                            mse_mean_day, mae_mean_day, mase_mean_day,
                            std_mse_mean_day,   std_mae_mean_day,  std_mase_mean_day,

                            "modl", classifier, file_name, nb_days,
                            file=open(glob_csv_res, "a"), sep=';')
                ###################################################
                #      E N D   C O C L U S T E R I N G            #
                ###################################################

            ###################################################
            #       D E B U G : C L U S T E R I N G           #
            ###################################################
            my_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 70, 100, 125, 150, 175, 200, 250, 300]
            my_range = [i for i in my_range if i < math.floor(nb_days * 0.75)] # Limitate number of clusters!

            for func, label, crange in [(cm.do_kmeans, "kmeans", my_range), (cm.do_kshape, "kshape", my_range), (cm.do_gak, "gak", my_range)]:
                valid_res = []
                for (fit, classifier) in [(fit_bayes, "bayes"), (fit_tree, "tree"), (fit_lr, "linear_reg")]:
                    logging.info("Loop: clustering %s with classifier %s for %s file", label, classifier, file_name)
                    try:
                        for nb_cluster in crange:
                            logging.info("nb_cluster is %s", nb_cluster)
                            e, v = wrapper.clustering_wrapper(
                                (func, "train_" + label),
                                days_train, days_valid, mean_day_train,
                                l_ref, file_name, nb_cluster,
                                fit, classifier)
                            valid_res.append((fit, classifier, nb_cluster, v))
                            if e:
                                logging.info("For file %s with %s + %s: found one empty cluster - break", file_name, label, classifier)
                                break
                    except IndexError as e:
                        logging.info("Loop: got IndexError using %s classifier and %s clustering", classifier, label)
                        pass

                for clab in ["bayes", "tree", "linear_reg"]:
                    res_val = [t for t in valid_res if t[1].startswith(clab)]
                    classifier = res_val[0][1]
                    fit = res_val[0][0]
                    logging.info("Exploit results: %s clab for clustering %s results", clab, label)

                    v_classifier, best_num_clust = utils.find_best_perf_by_index(res_val, 301)
                    logging.info("best_num_clust is %s", best_num_clust)

                    # Intermediate results storage
                    (
                        mse_non_proba, mse_proba, mae_non_proba,
                        mae_proba, mase_non_proba, mase_proba) = v_classifier[3][0]
                    (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or) = v_classifier[3][1]
                    (mse_mean_day, mae_mean_day, mase_mean_day) = v_classifier[3][2]
                    classifier_acc = v_classifier[3][3]
                    (
                        std_mse_non_proba, std_mse_proba, std_mse_mean_day,
                        std_mae_non_proba, std_mae_proba, std_mae_mean_day,
                        std_mase_non_proba, std_mase_proba, std_mase_mean_day) = v_classifier[3][4]
                    (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or) = v_classifier[3][5]

                    print(
                        mse_ecml_colin,  mse_ecml_fake,  mse_ecml_mean,
                        mae_ecml_colin,  mae_ecml_fake,  mae_ecml_mean,
                        mase_ecml_colin, mase_ecml_fake, mase_ecml_mean,
                        std_mse_colin, std_mse_fake, std_mse_mean,
                        std_mae_colin, std_mae_fake, std_mae_mean,
                        std_mase_colin, std_mase_fake, std_mase_mean,
                        mse_ar_error, mae_ar_error, mase_ar_error,
                        best_num_of_kmean_ecml,

                        classifier_acc,
                        best_num_clust, -1, -1,

                        mse_non_proba, mae_non_proba, mase_non_proba,
                        std_mse_non_proba, std_mae_non_proba, std_mase_non_proba,

                        mse_proba, mae_proba, mase_proba,
                        std_mse_proba, std_mae_proba, std_mase_proba,

                        mse_non_proba_or, mae_non_proba_or, mase_non_proba_or,
                        std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or,

                        mse_mean_day, mae_mean_day, mase_mean_day,
                        std_mse_mean_day,   std_mae_mean_day,  std_mase_mean_day,

                        label, classifier, file_name, nb_days,
                        file=open(vali_csv_res, "a"), sep=';')

                    while True:
                        logging.info("best_num_clust is %s", best_num_clust)
                        try:
                            _, res_clus = wrapper.clustering_wrapper(
                                (func, "train_val_" + label),
                                days_train_and_valid, days_test, mean_day_train_and_valid,
                                l_ref, file_name, best_num_clust,
                                fit, classifier)
                        except Exception as e:
                            logging.info("Exploit results: got exception using %s classifier and %s clustering", classifier, label)
                            best_num_clust = best_num_clust - 20
                            continue
                        break

                    # Store final results
                    (mse_non_proba, mse_proba, mae_non_proba, mae_proba, mase_non_proba, mase_proba) = res_clus[0]
                    (mse_non_proba_or, mae_non_proba_or, mase_non_proba_or) = res_clus[1]
                    (mse_mean_day, mae_mean_day, mase_mean_day) = res_clus[2]
                    classifier_acc = res_clus[3]
                    (std_mse_non_proba, std_mse_proba, std_mse_mean_day, std_mae_non_proba, std_mae_proba, std_mae_mean_day, std_mase_non_proba, std_mase_proba, std_mase_mean_day) = res_clus[4]
                    (std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or) = res_clus[5]

                    print(
                        mse_ecml_colin,  mse_ecml_fake,  mse_ecml_mean,
                        mae_ecml_colin,  mae_ecml_fake,  mae_ecml_mean,
                        mase_ecml_colin, mase_ecml_fake, mase_ecml_mean,
                        std_mse_colin, std_mse_fake, std_mse_mean,
                        std_mae_colin, std_mae_fake, std_mae_mean,
                        std_mase_colin, std_mase_fake, std_mase_mean,
                        mse_ar_error, mae_ar_error, mase_ar_error,
                        best_num_of_kmean_ecml,

                        classifier_acc,
                        best_num_clust, -1, -1,

                        mse_non_proba, mae_non_proba, mase_non_proba,
                        std_mse_non_proba, std_mae_non_proba, std_mase_non_proba,

                        mse_proba, mae_proba, mase_proba,
                        std_mse_proba, std_mae_proba, std_mase_proba,

                        mse_non_proba_or, mae_non_proba_or, mase_non_proba_or,
                        std_mse_non_proba_or, std_mae_non_proba_or, std_mase_non_proba_or,

                        mse_mean_day, mae_mean_day, mase_mean_day,
                        std_mse_mean_day,   std_mae_mean_day,  std_mase_mean_day,

                        label, classifier, file_name, nb_days,
                        file=open(glob_csv_res, "a"), sep=';')
            else:
                logging.info("Coclustering algorithms did not use 'n_day_' dimention to cut the cube: "
                                "cannot process coclus further for this file.")
        except Exception as inst:
            logging.error("An exception has occured.")
            logging.error(inst)
            pass

# Save code and data I use for this particular expe
fh.zip_data(in_dir, out_path_res) 
logging.info("Exec is finished without being interupted")
