import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import compute_features

@pytest.fixture
def sample_stock_data():
    """Creates a sample DataFrame for testing."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='min'))
    symbols = ['AAPL'] * 100
    index = pd.MultiIndex.from_tuples(zip(symbols, dates), names=['symbol', 'timestamp'])
    
    data = {
        'open': np.random.uniform(150, 152, 100),
        'high': np.random.uniform(152, 154, 100),
        'low': np.random.uniform(148, 150, 100),
        'close': np.random.uniform(150, 154, 100),
        'volume': np.random.randint(10000, 20000, 100)
    }
    return pd.DataFrame(data, index=index)

def test_compute_features_runs_and_adds_columns(sample_stock_data):
    """
    Tests that the compute_features function runs and adds the expected columns.
    """
    featured_df = compute_features(sample_stock_data)
    
    expected_cols = [
        'return_1m', 'mom_5m', 'mom_15m', 'mom_60m', 
        'vol_15m', 'vol_60m', 'vol_zscore', 
        'time_sin', 'time_cos'
    ]
    
    # Check that the function returns a DataFrame
    assert isinstance(featured_df, pd.DataFrame)
    
    # Check that all expected columns were added
    for col in expected_cols:
        assert col in featured_df.columns
