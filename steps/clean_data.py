import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing import Annotated
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStrategy

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"] 
]:
    """
    Clean the data by removing duplicates and handling missing values.
    Args:
        df (pd.DataFrame): DataFrame containing the ingested data
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple containing the cleaned data
    """
    try:
        logging.info("Cleaning data...")
        process_strategy=DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy=DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed.")
        return X_train, X_test, y_train, y_test
       
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e