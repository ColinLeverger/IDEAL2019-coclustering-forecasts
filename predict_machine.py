# Import lib
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Statistics tools
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from clustering_machine import ClusteringMachine
from hmm_machine import HmmMachine
from my_utils import *

class PredictMachine:

    def __init__(self): 
        self.clust_machine = ClusteringMachine()
        self.hmm_machine = HmmMachine()
        self.utils_cl = MyUtils()
        logging.info("Predict machine instantiated")

    @staticmethod
    def predict_median_hmm(class_of_curr_day, trans_mat, km, real_class_of_following_day = -1):
        """
        Find the most likely cluster for day n+1 and return its median as a
        prediction.

        Parameters
        ----------
        class_of_curr_day: int representing class of next day 
        trans_mat: HMM
        transition mat km: kmeans pre-trained algorithm
        :param class_of_curr_day:
        :param trans_mat:
        :param km:
        :param real_class_of_following_day:
        """
        if real_class_of_following_day == -1:
            return km.cluster_centers_[trans_mat.loc[class_of_curr_day].idxmax(1)]
        else:
            return km.cluster_centers_[real_class_of_following_day]

    def do_forecast_ar_model(self, today, train, test):
        # train autoregression
        model_fit = AR(train.fillna(0)).fit()
        logging.info("Fitted AR...")

        AResults = model_fit.predict(
            start = len(train), end = len(train) + len(test) - 1)
        logging.info("Predicted AR")

        mse = self.utils_cl.compute_mse(test, AResults)
        mae = self.utils_cl.compute_mae(test, AResults)
        mase = self.utils_cl.compute_mase(today, test, AResults)

        logging.info("Exit do_forecast_ar_model")
        return AResults, mse, mae, mase

    def do_forecast_hw_model(self, today, train, test, sp):
        logging.info("Fitting ExponentialSmoothing...")
        fit = ExponentialSmoothing(
            np.asarray(train), seasonal_periods=sp,
            trend='add', seasonal='add').fit()
        logging.info("Fitted ExponentialSmoothing.")
        hw = fit.predict(start = len(train), end = len(train) + len(test) - 1)
        logging.info("Predicted ExponentialSmoothing")

        error = self.utils_cl.compute_mse(test, hw)
        mae = self.utils_cl.compute_mae(test, hw)
        mase = self.utils_cl.compute_mase(today, test, hw)
        
        logging.info("Exit do_forecast_hw_model")
        return hw, error, mae, mase
