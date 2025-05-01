import logging
from zenml import step
import pandas as pd

@step

def evaluate_model(df:pd.DataFrame) -> None:
    """
    Evaluate the model using the provided test data.
    Args:
        model: The trained model to be evaluated
        test_data: DataFrame containing the test data
    Returns:
        None
    """
    pass
   