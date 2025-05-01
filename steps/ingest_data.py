import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """
        Initialize the IngestData class with the path to the data file.
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting th data from the given path
    Args:data_path (str): Path to the data file
    Returns:
        pd.DataFrame: DataFrame containing the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e