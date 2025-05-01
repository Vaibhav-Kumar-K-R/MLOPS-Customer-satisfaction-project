import logging
import pandas as pd
from zenml import step

@step

def train_model(df: pd.DataFrame) -> None:
    """
    Train the model using the provided DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the cleaned data
    Returns:
        None
    """
    pass

    