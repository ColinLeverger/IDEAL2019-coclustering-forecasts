# Import lib
import warnings
from my_utils import MyUtils
import logging

# Visualisations
import matplotlib as mpl

# Clustering 
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
from tslearn.clustering import GlobalAlignmentKernelKMeans

# tslearn
from tslearn.utils import to_time_series_dataset

warnings.simplefilter("error", RuntimeWarning)

# Configure size of figures
mpl.rcParams['figure.figsize'] = (20, 20)


class ClusteringMachine:

    def __init__(self):
        self.utils_cl = MyUtils()

        logging.info("Clustering machine instantiated")

    def do_kmeans_wrapper(self, df, km_size):
        """ 
        Do a kmeans for a df with several columns, knowing: 
          * wanted cluster number, 
          * size of the data sample needed needed.
        """
        (km, y_pred) = self.do_kmeans(df, km_size)

        cc = self.create_c_a_v(df, y_pred)

        return km, cc

    @staticmethod
    def format_clust_results(nb_cluster, cc):
        cluster_and_values = dict((str(k), []) for k in range(1, nb_cluster + 1))
        zips_kmeans = []
        for e in cc:
            cluster_and_values[str(e[1] + 1)].append(e[0])
            zips_kmeans.append((str(e[1] + 1), e[0], 24, 1))
        return cluster_and_values, zips_kmeans

    @staticmethod
    def do_kmeans(days, km_size):
        """
        From a time series (as a list of df called days), creates km_size 
        clusters using kmeans algo.

        Parameters
        ----------
          * days: time series to cluster 
          * km_size: number of clusters needed 

        Returns
        ----------
          * km: k-means object generated for the clustering, it contains info about the algorithm
          * y_pred: results of the clustering, it contains the clusters themselves
        """
        # Arrange data for our lib
        unq = days["n_day_"].unique()
        values = [days[days["n_day_"] == l]["val_"].values for l in unq]
        formatted_dataset = to_time_series_dataset(values)

        # Configure our kmeans
        km = TimeSeriesKMeans(
            n_clusters=km_size,
            metric="euclidean",
            random_state=42,
            verbose=False)

        y_pred = km.fit_predict(formatted_dataset)

        return km, y_pred

    @staticmethod
    def do_kshape(days, km_size):
        """
        From a time series (as a list of df called days), creates km_size 
        clusters using kshape algo.
        """
        # Arrange data for our lib
        unq = days["n_day_"].unique()
        values = [days[days["n_day_"] == l]["val_"].values for l in unq]
        formatted_dataset = to_time_series_dataset(values)

        # Configure our kmeans
        kshape = KShape(
            n_clusters=km_size,
            random_state=42,
            verbose=False)

        y_pred = kshape.fit_predict(formatted_dataset)

        return kshape, y_pred

    @staticmethod
    def do_gak(days, km_size):
        """
        From a time series (as a list of df called days), creates km_size 
        clusters using GAK algo.
        """
        # Arrange data for our lib
        unq = days["n_day_"].unique()
        values = [days[days["n_day_"] == l]["val_"].values for l in unq]
        formatted_dataset = to_time_series_dataset(values)

        # Configure our kmeans
        gak = GlobalAlignmentKernelKMeans(
            n_clusters=km_size, verbose=False, random_state=42
        )

        y_pred = gak.fit_predict(formatted_dataset)

        return gak, y_pred

    @staticmethod
    def create_c_a_v(days, y_pred):
        """
        From days and results (predictions) of kmeans, create specific data
        format tbused afterwards on scripts.

        Parameters
        ----------
          * days: time series that has been clustered
          * y_pred: prediction generated 

        Returns
        ----------
          * cc: clusters generated but in a comprehensive format (array(("colname1", cluster_num_1), ("colname2", cluster_num_2), ...))
        """
        # HACK: This colnames_track is used to keep a track of sequences
        # of days. this is dirty but necessary.         
        # colnames_track: track of colnames for cluster, ordered 
        #   (array("colname1", "colname2", ...)) 
        colnames_track = days["n_day_"].unique()

        # Make results interpretable
        cc = sorted(
            list(zip(colnames_track, y_pred)), key=lambda tup: tup[1]
        )

        return cc

    @staticmethod
    def apply_clustering(days, clust):
        """
        Apply given clustering algorithm on given dataset.
        """
        # Arrange data for our lib
        unq = days["n_day_"].unique()
        values = [days[days["n_day_"] == l]["val_"].values for l in unq]
        formatted_dataset = to_time_series_dataset(values)

        y_pred = clust.predict(formatted_dataset)

        return y_pred
