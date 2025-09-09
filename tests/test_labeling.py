import pandas as pd
import numpy as np
import pytest
from src.labeling import create_labels

@pytest.fixture
def sample_featured_data():
    """Creates a sample DataFrame with a 'close' column for testing labels."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='min'))
    symbols = ['AAPL'] * 10
    index = pd.MultiIndex.from_tuples(zip(symbols, dates), names=['symbol', 'timestamp'])
    
    # Create a predictable price movement
    close_prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 100]
    data = {'close': close_prices}
    return pd.DataFrame(data, index=index)

def test_create_labels_correctly_classifies(sample_featured_data):
    """
    Tests that the create_labels function correctly assigns 1, -1, and 0.
    """
    # Using a large horizon to get clear movements
    labeled_df = create_labels(sample_featured_data, horizon=2, threshold=0.005)
    
    # Expected labels based on the price movement and a 0.5% threshold
    # 0: 102/100 - 1 = 0.02 -> 1
    # 1: 101/101 - 1 = 0.0 -> 0
    # 2: 100/102 - 1 = -0.019 -> -1
    # 3: 99/101 - 1 = -0.019 -> -1
    # 4: 98/100 - 1 = -0.02 -> -1
    # 5: 99/99 - 1 = 0.0 -> 0
    # 6: 100/98 - 1 = 0.02 -> 1
    # 7: 100/99 - 1 = 0.01 -> 1
    expected_labels = [1, 0, -1, -1, -1, 0, 1, 1, np.nan, np.nan]
    
    # Drop NaNs from both expected and actual results for comparison
    actual_labels = labeled_df['label'].dropna()
    expected_labels = [e for e in expected_labels if not pd.isna(e)]

    assert actual_labels.tolist() == expected_labels
