import logging
import pandas as pd
import numpy as np
from zenml import step
import pandas as pd
from src.evaluation import MSE,RMSE,R2
from sklearn.base import RegressorMixin
from typing import Annotated
from typing import Tuple
from zenml.client import Client
import mlflow 

experiment_tracker=Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)

def evaluate_model(model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.DataFrame) -> Tuple[
    Annotated[float,"r2_score"],
    Annotated[float,"rmse"]
]:
    """
    Evaluate the model using the provided test data.
    Args:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The test features.
        y_test (pd.DataFrame): The true labels for the test data.
    """
    try:

        prediction=model.predict(X_test)
        mse_class=MSE()
        mse=mse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("mse", mse)
        r2_class=R2()
        r2_score=r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("r2_score", r2_score)
        rmse_class=RMSE()
        rmse=rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse", rmse)
        return r2_score,rmse
    except Exception as e:
        logging.error("Error in evaluation model".format(e))
        raise e 



   