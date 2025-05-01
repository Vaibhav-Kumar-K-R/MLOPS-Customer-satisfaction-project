import logging
from zenml import step
import pandas as pd

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing duplicates and handling missing values.
    Args:
        df (pd.DataFrame): DataFrame containing the ingested data
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        logging.info("Cleaning data...")
        # Remove duplicates
      
        return df
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e