import logging
import os
import platform
import sys
import json
import numpy as np
import pandas as pd
from my_utils import MyUtils
from file_helper import FileHelper

# Check the Khiops python directory on linux
if platform.system() == "Linux" :
    khiopsPythonDir = os.path.join(os.environ["HOME"], "pykhiops")
    if not os.path.isdir(khiopsPythonDir):
        raise ValueError("The pykhiops library path is not correctly set. Copy the library to " + khiopsPythonDir)

# Specify Khiops python lib path
khiopsLibPath = os.path.join(os.environ["KhiopsHome"], "python", "lib") if platform.system() == "Windows" else os.path.join(os.environ["HOME"], "pykhiops", "lib")
if not khiopsLibPath in sys.path:
    sys.path.append(khiopsLibPath)

# Import pykhiops
import pykhiops as pk
from pykhiops.coclusteringResults import CoclusteringResults

# Khiops sample directory: the sample function exploit the Adult sample dataset of Khiops
if not os.path.isdir(os.path.join(pk.getKhiopsSampleDir(), "Adult")):
    raise ValueError("Adult Samples directory does not exist in " + pk.getKhiopsSampleDir())


class KhiopsManager:

    def __init__(self):
        logging.info(pk.getKhiopsInfo())
        self.this_file_dir   = os.path.dirname(os.path.realpath(__file__))

        # path mgmt
        # use timestamp of each exec in paths
        self.fh = FileHelper()
        self.dictionary_file = os.path.join(
            self.this_file_dir, "dic", "series.kdic")        
        self.classif_res = os.path.join(
            self.this_file_dir, "res", "khiops_res", "classif")
        self.coclus_res = os.path.join(
            self.this_file_dir, "res", "khiops_res", "coclus")     
        self.pred_res = os.path.join(
            self.this_file_dir, "res", "khiops_res", "pred_res")    

        self.fh.ensure_dirs_exist(
            [
                self.dictionary_file, self.classif_res, 
                self.coclus_res, self.pred_res
            ])

        self.ccr = CoclusteringResults()
        self.utils = MyUtils()
        logging.info("Khiops manager instantiated")
        logging.info("dictionary_file used: %s", self.dictionary_file)

    """KHIOPS COCLUSTERING TRAIN AND SIMPLIFICATIONS"""
    def train_coclustering(self, f):
        """
        Train a coclustering model in the simplest way possible
        """
        file_name = self.fh.get_file_name(f)
        logging.info("Train of coclustering for file %s", file_name)

        # Train coclustering model for variables "SampleId" and "Char"
        pk.trainCoclustering(
            dictionaryFile = self.dictionary_file, 
            dictionary = "train", dataTable = f,
            coclusteringVariables = ["time_", "n_day_", "val_"], 
            resultsDir = self.coclus_res, 
            fieldSeparator = ";", samplePercentage = 100, 
            resultsPrefix = file_name + "_")
    
    def simplify_coclustering(self, file_name, mpi = 100, mcn = 999999):
        """
        Simplify a coclustering model in the simplest way possible
        """
        base_path = os.path.join(self.coclus_res, file_name)
        self.fh.ensure_dirs_exist([base_path])
        cf = os.path.join(self.coclus_res, file_name + "_Coclustering.khc" )  
        logging.info("Simplify coclustering for file %s", file_name)
        logging.info("MCN=%s, MPI=%s", mcn, mpi)

        if mcn != 999999:
            scf = file_name + "_Simplified-" + str(mcn) + ".khc"
        else:
            scf = file_name + "_Simplified-" + str(mpi) + ".khc"

        logging.info("scf=%s", scf)
        
        pk.simplifyCoclustering(
            coclusteringFile = cf, simplifiedCoclusteringFile = scf, 
            resultsDir = base_path,
            maxCellNumber = mcn, 
            maxPreservedInformation = mpi, 
        )

        return str(os.path.join(base_path, scf)).replace(".khc", ".json")
    
    def deploy_coclustering(self, file_name, identifier):
        """
        Deploy a coclustering model
        """
        base_path = os.path.join(self.coclus_res, file_name)
        self.fh.ensure_dirs_exist([base_path])

        logging.info("Deploy coclustering for file %s", file_name)
        scf = os.path.join(self.coclus_res, file_name, file_name + "_Simplified-" + str(identifier) + ".khc")
        
        dep_prefix = file_name +"_Deployed-" + str(identifier) + "-"

        pk.prepareCoclusteringDeployment(
            dictionaryFile = self.dictionary_file, 
            dictionary = "root", coclusteringFile = scf,
            tableVariable = "secondary",
            deployedVariable = "n_day_",
            buildDistanceVariables = True,
            resultsPrefix = dep_prefix,
            resultsDir = base_path
        )
        return os.path.join(base_path, dep_prefix + "Coclustering.kdic") 

    def transfer_database(self, d, deployed_path, f, identifier):
        """
        Transfer a coclustering model to use it
        """
        file_name = self.fh.get_file_name(f) 
        out_path = os.path.join(
            self.coclus_res, file_name, file_name + "_Transf-" + str(identifier) + ".csv")
            
        logging.info("Transfering of coclustering for file %s into %s", file_name, out_path)

        pk.transferDatabase( 
            dictionaryFile = d,
            dictionary = "root",
            dataTable = deployed_path,
            additionalDataTables = {"root`secondary": f},
            fieldSeparator = ";", 
            outputFieldSeparator = ";", 
            outputDataTable = out_path)
        return out_path

    """JSON COCLUSTERING MANIPULATIONS"""
    def get_clusters(self, path):
        """
        From a given Khiops json file, extract clusters, and detailed info about
        them.

        Returns:
          * zips: more info about each cluster zipped together (e.g all tuples that goes together 'aligned') => tuples (clus, values, value_frequencies, value_typicalities)
          * cluster_and_values: dic(cluster_num => values_associated)
        """
        cluster_and_values = {}
        zips = []
        with open(path) as f:
            data = json.load(f)["coclusteringReport"]["dimensionPartitions"]
            n_days = list(filter(lambda d: d['name'] in "n_day_", data))

            for idx, n_day in enumerate(n_days[0]["valueGroups"], start = 1):
                # Ugly turnaround to manipulate floats and int
                # see https://goo.gl/8tYfhn
                values = list(map(int, map(float, n_day["values"])))              
                clus   = n_day["cluster"]
                value_frequencies  = n_day["valueFrequencies"]
                value_typicalities = n_day["valueTypicalities"]

                zips.append(list(
                    zip(clus, values, value_frequencies, value_typicalities)))
                cluster_and_values[clus] = values
        return self.utils.flattify(zips), cluster_and_values

    @staticmethod
    def get_clusters_from_dep(path):
        """
        From a given Khiops transferred thing, extract clusters, and detailed
        info about them.

        Returns:
          * cluster_and_values: dic(cluster_num => values_associated)
        """
        with open(path) as f:
            df = pd.read_csv(f, sep = ";")
            
            k = df["n_day_PredictedLabel"].unique()
            cluster_and_values =  dict((key, []) for key in k)

            for index, row in df.iterrows():     
                n_day_ = row["n_day_"]
                clus  = row["n_day_PredictedLabel"]
                cluster_and_values[clus].append(n_day_)
        return cluster_and_values

    def get_cells_number(self, file_name, mpi = None):
        """
        From a given Khiops json file, extract number of cells for all
        dimentions

        return:
          - cells
        """
        if mpi is not None:
            p = os.path.join(self.coclus_res, file_name, file_name + "_Simplified-" + str(mpi)+ ".json")
        else:
            p = os.path.join(self.coclus_res, file_name + "_Coclustering.json")
        with open(p) as f:
            cells = json.load(f)["coclusteringReport"]["summary"]["cells"]

        return int(cells)

    def get_cluster_number(self, file_name, ref_id = None):
        """
        From a given Khiops json file, extract number of cluster for dim
        "n_day_" 

        return:
          - nb_of_cluster_found
        """  
        if ref_id is not None:
            p = os.path.join(self.coclus_res, file_name, file_name + "_Simplified-" + str(ref_id) + ".json")
        else:
            p = os.path.join(self.coclus_res, file_name + "_Coclustering.json") 
        with open(p) as f:
            data = json.load(f)["coclusteringReport"]["dimensionSummaries"]
            n_days = list(filter(lambda d: d['name'] in "n_day_", data))

        if not n_days:
            return False
        else:
            return int(n_days[0]["parts"])

    """UTILS"""
    @staticmethod
    def _compute_accuracy(len_y_pred, y_pred, y_pred_target):
        """
        Compute the accuracy of a classifier.
        """
        c = 0
        for i in range(0, len_y_pred): 
            pred_group_day_ahead = str(y_pred.iloc[i]["Predictedy"])
            real_group_day_ahead = str(y_pred_target.values[i])
            if pred_group_day_ahead != real_group_day_ahead: 
                c = c + 1
        train_acc = 100 - c * 100 / len_y_pred
        logging.debug("%s errors over %s values", c, len_y_pred)
        logging.debug("That is %s perc of accuracy", train_acc)
        return train_acc

    @staticmethod
    def __get_typicalitie(n_day, my_zip):
        """
        From a list of tuples my_zip, find the
        tuple which concern n_day and retrieve its valTyp.

        PARAMETERS
        ------------------------
        - my_zip: tuples (group, n_day, valFreq, valTyp) 
        - n_day: int
        """
        items = [i for i in my_zip if n_day == i[1]]
        return items[0][3]

    @staticmethod
    def compute_centroids(c_a_v, df, l_ref):
        """
        This method computes mean days and centroids for given values

        PARAMETERS
        ------------------------
          * c_a_v: dic(cluster_num => values_associated)
          * df: dataset studied
          * l_ref: len of one ref day

        RETURNS
        -------------------------
          * centroids: dic(cluster_id: int => centroid: [])
        """
        centroids = {}
        e = False
        for k, v in c_a_v.items():
            # If there is at least one day in the cluster considered!
            if len(v) != 0: 
                logging.info("Computing cluster %s centroid", k)                
                centroids[k] = df[df['n_day_'].isin(map(str, v))].groupby('time_')['val_'].agg('mean').values
            # Otherwise, mean day = 0 
            else:
                logging.info("Cluster %s is empty", k)
                e = True
                centroids[k] = [0] * l_ref
        logging.info("Keys of centroids are %s", list(centroids.keys()))
        logging.debug("Centroids are %s", centroids)
        return centroids, e

    def process_pred_proba(self, 
                        y_pred, y_pred_target, clusts_names,
                        centroids, days_to_predict,  
                        nb_cluster_found, n_pt_per_day, 
                        mean_day, o = False):
        """
        This method process results from Khiops classification method to create
        predictions and compute mses. If Oracle mode is "True", then use a mock
        for classifier Y (e.g the classifier knows exactly the Y columns)

        WHAT IT DOES
        ------------------------
          * First thing (A): Compute given classifier accuracy regarding known 
            y_pred_target. 
          * Second thing (B): Process probalistic classifier results but not
            using the probabilities and only the most probable class for day n +
            1. NPA
          * Third thing (C): Process probalistic classifier results using
            probabilities affected to each cluster (e.g. day n + 1 could be 10%
            in cluster 1, 50% in cluster 2, etc). Use ponderations to sum up
            each centroids and produce result. PA
          * Last thing (D): Wrap up, means computations => compute results...

        PARAMETERS
        ------------------------
          * y_pred: Matrix results from classifier
          * y_pred_target_test: Known y_pred_target for classifier for test data
          * clusts_names: names of all clusters
          * centroids: clusters centroids
          * days_to_predict: list of days that have been used for test
          * nb_cluster_found: value exctracted from Khiops coclustering file;
            used to parameterize loops and computations.
          * n_pt_per_day: if there is one point per hour of the day, then this
            value will be 24.
          * mean_day: pre-computed mean day for all training ensemble.
          * oracle: are we in oracle mode? ('Y' known)

        RETURN
        -------------------------
          * Tuple(mean_mse_non_proba, mean_mse_proba, classifier_acc,
            mean_mse_mean_day) => Meaned MSE !
            :param y_pred:
            :param clusts_names:
            :param centroids:
            :param days_to_predict:
            :param nb_cluster_found:
            :param n_pt_per_day:
            :param mean_day:
            :param y_pred_target:
            :param o:
        """
        len_y_pred = len(y_pred)

        # (A): Compute classifier accuracy
        classifier_acc = self._compute_accuracy(len_y_pred, y_pred, y_pred_target)
        if classifier_acc == 0.0:
            logging.info("This classifier is very not good and failed all the time")

        # Init empty arrays
        mses_non_proba = []
        mses_proba = []    
        mses_mean_day = [] 

        mae_non_proba = []
        mae_proba = []    
        mae_mean_day = [] 

        mase_non_proba = []
        mase_proba = []    
        mase_mean_day = [] 

        # Think about removing last day, which has not days after
        n_days_ = days_to_predict["n_day_"].unique()[:-1] 
        # Also got to remove the first day, which has not day before
        # Wrangling data to have good types.
        n_days_int = list(map(int, n_days_))
        n_days_int.remove(min(n_days_int))
        n_days_ = list(map(str, n_days_int))

        for ref_in_classif_ensemble, z in enumerate(n_days_):
            """
            (B): NPA approach
            """
            indice_today = str(int(z) - 1)
            today_vals = days_to_predict[days_to_predict["n_day_"] == indice_today]["val_"].values
            tomor_vals = days_to_predict[days_to_predict["n_day_"] == z]["val_"].values
            tom_class_pred_cluster_y = y_pred.iloc[ref_in_classif_ensemble]["Predictedy"]
            tom_pred_y = centroids[str(tom_class_pred_cluster_y)]

            # Append mses
            mses_non_proba.append(
                self.utils.compute_mse(tomor_vals, tom_pred_y))
            mses_mean_day.append(
                self.utils.compute_mse(tomor_vals, mean_day))
            
            mae_non_proba.append(
                self.utils.compute_mae(tomor_vals, tom_pred_y))
            mae_mean_day.append(
                self.utils.compute_mae(tomor_vals, mean_day))
            
            mase_non_proba.append(
                self.utils.compute_mase(today_vals, tomor_vals, tom_pred_y))
            mase_mean_day.append(
                self.utils.compute_mase(today_vals, tomor_vals, mean_day))

            mean_days_pond = []        
            p_tot = 0
            """
            (C): PA approach
            """
            for c in clusts_names:
                try:
                    # Get probability to be in groupe number "c"
                    p = y_pred.iloc[ref_in_classif_ensemble]["Proby" + str(c)]

                    # p_tot variable for debug purposes                
                    p_tot = p_tot + p
                except:
                    logging.debug("Did not find %s in the prediction matrix", c) 
                    p = 0
                    pass

                # Retrieve mean day for cluster c
                day_pond = centroids[c].copy()

                # Scale according to proba
                day_pond[:] = [x * p for x in day_pond]
                mean_days_pond.append(day_pond)

            logging.debug("p_tot is %s", p_tot)  # Here it should be 100. 

            # Sum up everything to create prediction  
            tomo_pred_with_pond = [0] * n_pt_per_day
            for d_p in mean_days_pond:
                for idx, e in enumerate(d_p):
                    tomo_pred_with_pond[idx] = tomo_pred_with_pond[idx] + e

            mses_proba.append(
                self.utils.compute_mse(tomor_vals, tomo_pred_with_pond))
            mae_proba.append(
                self.utils.compute_mae(tomor_vals, tomo_pred_with_pond))
            mase_proba.append(
                self.utils.compute_mase(today_vals, tomor_vals, tomo_pred_with_pond))

        # (D): Wrap up: mean MSEs (so we have, at the end, Meaned Mean Square
        # Error)
        # MMSES
        mean_mse_non_proba = np.mean(mses_non_proba)
        mean_mse_proba     = np.mean(mses_proba)
        mean_mse_mean_day  = np.mean(mses_mean_day)

        mean_mae_non_proba = np.mean(mae_non_proba)
        mean_mae_proba     = np.mean(mae_proba)
        mean_mae_mean_day  = np.mean(mae_mean_day)

        mean_mase_non_proba = np.mean(mase_non_proba)
        mean_mase_proba     = np.mean(mase_proba)
        mean_mase_mean_day  = np.mean(mase_mean_day)

        # MSTD
        std_mse_non_proba = np.std(mses_non_proba)
        std_mse_proba     = np.std(mses_proba)
        std_mse_mean_day  = np.std(mses_mean_day)

        std_mae_non_proba = np.std(mae_non_proba)
        std_mae_proba     = np.std(mae_proba)
        std_mae_mean_day  = np.std(mae_mean_day)

        std_mase_non_proba = np.std(mase_non_proba)
        std_mase_proba     = np.std(mase_proba)
        std_mase_mean_day  = np.std(mase_mean_day)

        return (
            (mean_mse_non_proba, mean_mse_proba, mean_mse_mean_day), 
            (mean_mae_non_proba, mean_mae_proba, mean_mae_mean_day), 
            (mean_mase_non_proba, mean_mase_proba, mean_mase_mean_day),
            classifier_acc,
            (std_mse_non_proba, std_mse_proba, std_mse_mean_day),
            (std_mae_non_proba, std_mae_proba, std_mae_mean_day),
            (std_mase_non_proba, std_mase_proba, std_mase_mean_day)
        )
