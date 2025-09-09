import pandas as pd
import numpy as np
from src.utils.logging_config import logger

def create_labels(df, horizon=5, threshold=0.0002):
    """
    Creates regression and classification labels for the given stock data.

    Args:
        df (pd.DataFrame): DataFrame with feature columns.
                           Must have a multi-index with ('symbol', 'timestamp').
        horizon (int): The prediction horizon in minutes.
        threshold (float): The threshold for classifying returns as up, down, or flat.

    Returns:
        pd.DataFrame: The DataFrame with added 'fwd_return' and 'label' columns.
    """
    logger.info(f"Creating labels with a {horizon}-minute horizon...")

    # Calculate the forward return
    df['fwd_return'] = df.groupby(level='symbol')['close'].transform(
        lambda x: x.shift(-horizon) / x - 1
    )

    # Create the classification label
    def classify_return(fwd_return):
        if fwd_return > threshold:
            return 1  # Up
        elif fwd_return < -threshold:
            return -1 # Down
        else:
            return 0  # Flat

    df['label'] = df['fwd_return'].apply(classify_return)
    
    logger.info("Label creation complete.")
    
    return df.dropna()
