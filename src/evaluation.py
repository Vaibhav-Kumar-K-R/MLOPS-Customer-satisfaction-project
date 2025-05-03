import logging
import numpy as np
from abc import ABC,abstractmethod
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error

class Evaluation(ABC):
    """
        Strategy for defining models
    """
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        """"
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
        Evaluation starategy that uses mean square error
    """
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("Mean squared error:"+str(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        

class R2(Evaluation):
    """
        Evaluation starategy that uses r2 score
    """
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating R2")
            mse=r2_score(y_true,y_pred)
            logging.info("R2 score: "+str(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating R2: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
        Evaluation starategy that uses root mean square error
    """
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating Root mean square error")
            rmse=root_mean_squared_error(y_true,y_pred)
            logging.info("Root mean square  score: "+str(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating Root mean square error: {}".format(e))
            raise e
        