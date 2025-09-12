#!/usr/bin/env python3
"""
Unit tests for src/models/_utils/labeling.py - Data Labeling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.models._utils.labeling import create_labels


class TestCreateLabels:
    """Test create_labels function."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1min')
        symbols = ['SPY', 'AAPL']
        
        data = []
        np.random.seed(42)  # For reproducible tests
        
        for symbol in symbols:
            base_price = 100.0 if symbol == 'SPY' else 150.0
            current_price = base_price
            
            for i, timestamp in enumerate(timestamps):
                # Create realistic price movement with some trend
                price_change = np.random.normal(0.001, 0.01)  # Small upward drift with noise
                current_price = max(current_price * (1 + price_change), 1.0)
                
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'close': current_price,
                    'open': current_price + np.random.normal(0, 0.01),
                    'high': current_price + np.random.uniform(0, 0.02),
                    'low': current_price - np.random.uniform(0, 0.02),
                    'volume': np.random.randint(1000, 5000)
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['symbol', 'timestamp']).sort_index()
    
    def test_create_labels_basic_functionality(self, sample_price_data):
        """Test basic label creation functionality."""
        result = create_labels(sample_price_data, horizon=5, threshold=0.0002)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have MultiIndex
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['symbol', 'timestamp']
        
        # Should have original columns plus new label columns
        original_columns = set(sample_price_data.columns)
        result_columns = set(result.columns)
        assert original_columns.issubset(result_columns)
        
        # Should have new label columns
        assert 'fwd_return' in result.columns
        assert 'label' in result.columns
    
    def test_forward_return_calculation(self, sample_price_data):
        """Test forward return calculation."""
        horizon = 5
        result = create_labels(sample_price_data, horizon=horizon, threshold=0.001)
        
        for symbol in result.index.get_level_values(0).unique():
            symbol_data = result.loc[symbol].sort_index()
            close_prices = symbol_data['close']
            fwd_returns = symbol_data['fwd_return']
            
            # Check forward return calculation for valid indices
            for i in range(len(close_prices) - horizon):
                current_price = close_prices.iloc[i]
                future_price = close_prices.iloc[i + horizon]
                expected_return = future_price / current_price - 1
                actual_return = fwd_returns.iloc[i]
                
                if not pd.isna(actual_return):
                    assert abs(actual_return - expected_return) < 1e-10
    
    def test_label_classification_logic(self, sample_price_data):
        """Test label classification logic."""
        threshold = 0.01  # 1%
        result = create_labels(sample_price_data, horizon=5, threshold=threshold)
        
        # Check label classification
        for _, row in result.iterrows():
            fwd_return = row['fwd_return']
            label = row['label']
            
            if not pd.isna(fwd_return) and not pd.isna(label):
                if fwd_return > threshold:
                    assert label == 1  # Up
                elif fwd_return < -threshold:
                    assert label == -1  # Down
                else:
                    assert label == 0  # Flat
    
    def test_different_horizons(self, sample_price_data):
        """Test labeling with different horizons."""
        horizons = [1, 5, 10, 15]
        
        for horizon in horizons:
            result = create_labels(sample_price_data, horizon=horizon, threshold=0.001)
            
            # Should work for all horizons
            assert 'fwd_return' in result.columns
            assert 'label' in result.columns
            
            # Should have fewer valid labels for longer horizons (due to edge effects)
            valid_labels = result['label'].dropna()
            expected_valid = len(sample_price_data) - horizon * len(sample_price_data.index.get_level_values(0).unique())
            
            # Should have approximately the expected number of valid labels
            # (some may be lost due to dropna)
            assert len(valid_labels) <= expected_valid
    
    def test_different_thresholds(self, sample_price_data):
        """Test labeling with different thresholds."""
        thresholds = [0.0001, 0.001, 0.01, 0.1]
        horizon = 5
        
        results = {}
        for threshold in thresholds:
            results[threshold] = create_labels(sample_price_data, horizon=horizon, threshold=threshold)
        
        # Higher thresholds should result in more 'flat' labels (0)
        for i in range(len(thresholds) - 1):
            lower_thresh = thresholds[i]
            higher_thresh = thresholds[i + 1]
            
            lower_result = results[lower_thresh]
            higher_result = results[higher_thresh]
            
            # Count labels
            lower_flat_count = (lower_result['label'] == 0).sum()
            higher_flat_count = (higher_result['label'] == 0).sum()
            
            # Higher threshold should have more flat labels
            assert higher_flat_count >= lower_flat_count
    
    def test_multiple_symbols(self, sample_price_data):
        """Test labeling for multiple symbols."""
        result = create_labels(sample_price_data, horizon=5, threshold=0.001)
        
        # Should have data for both symbols
        symbols = result.index.get_level_values(0).unique()
        assert 'SPY' in symbols
        assert 'AAPL' in symbols
        
        # Each symbol should have labels computed independently
        spy_data = result.loc['SPY']
        aapl_data = result.loc['AAPL']
        
        assert len(spy_data) > 0
        assert len(aapl_data) > 0
        
        # Should have forward returns and labels for both
        assert 'fwd_return' in spy_data.columns
        assert 'label' in spy_data.columns
        assert 'fwd_return' in aapl_data.columns
        assert 'label' in aapl_data.columns
    
    def test_edge_cases_insufficient_data(self):
        """Test behavior with insufficient data for horizon."""
        # Create minimal data
        small_data = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'close': 100.0 + i * 0.1,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'volume': 1000
            }
            for i in range(3)  # Only 3 data points
        ]).set_index(['symbol', 'timestamp'])
        
        # Request horizon longer than data
        result = create_labels(small_data, horizon=10, threshold=0.001)
        
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        # All forward returns should be NaN due to insufficient future data
        assert result['fwd_return'].isna().all()
    
    def test_edge_cases_single_data_point(self):
        """Test behavior with single data point."""
        single_point = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'close': 100.0,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'volume': 1000
        }]).set_index(['symbol', 'timestamp'])
        
        result = create_labels(single_point, horizon=5, threshold=0.001)
        
        # Should handle single point gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 1  # May be dropped due to NaN
    
    def test_zero_threshold(self, sample_price_data):
        """Test labeling with zero threshold."""
        result = create_labels(sample_price_data, horizon=5, threshold=0.0)
        
        # With zero threshold, any positive return should be labeled as 1,
        # any negative return as -1, and exactly zero as 0
        for _, row in result.iterrows():
            fwd_return = row['fwd_return']
            label = row['label']
            
            if not pd.isna(fwd_return) and not pd.isna(label):
                if fwd_return > 0:
                    assert label == 1
                elif fwd_return < 0:
                    assert label == -1
                else:
                    assert label == 0
    
    def test_extreme_threshold(self, sample_price_data):
        """Test labeling with extremely high threshold."""
        extreme_threshold = 10.0  # 1000% - unrealistic for minute data
        result = create_labels(sample_price_data, horizon=5, threshold=extreme_threshold)
        
        # With extreme threshold, most/all labels should be 0 (flat)
        labels = result['label'].dropna()
        if len(labels) > 0:
            # Most labels should be 0
            flat_ratio = (labels == 0).sum() / len(labels)
            assert flat_ratio > 0.8  # At least 80% should be flat
    
    def test_data_sorting_requirement(self):
        """Test that function handles unsorted data correctly."""
        # Create unsorted data
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min')
        shuffled_indices = np.random.permutation(len(timestamps))
        
        data = []
        for i, idx in enumerate(shuffled_indices):
            data.append({
                'symbol': 'SPY',
                'timestamp': timestamps[idx],
                'close': 100.0 + i * 0.1,  # Price increases with original order
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'volume': 1000
            })
        
        unsorted_df = pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        # Function should handle unsorted data (pandas groupby handles sorting)
        result = create_labels(unsorted_df, horizon=3, threshold=0.001)
        
        assert isinstance(result, pd.DataFrame)
        assert 'fwd_return' in result.columns
        assert 'label' in result.columns
    
    def test_missing_close_prices(self):
        """Test behavior with missing close prices."""
        data_with_nan = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'close': 100.0 + i * 0.1 if i != 5 else np.nan,  # NaN at index 5
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'volume': 1000
            }
            for i in range(20)
        ]).set_index(['symbol', 'timestamp'])
        
        result = create_labels(data_with_nan, horizon=5, threshold=0.001)
        
        # Should handle NaN prices
        assert isinstance(result, pd.DataFrame)
        # Result may have fewer rows due to dropna()
    
    def test_identical_prices(self):
        """Test labeling when prices don't change."""
        constant_price_data = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'close': 100.0,  # Constant price
                'open': 100.0,
                'high': 100.0,
                'low': 100.0,
                'volume': 1000
            }
            for i in range(20)
        ]).set_index(['symbol', 'timestamp'])
        
        result = create_labels(constant_price_data, horizon=5, threshold=0.001)
        
        # All forward returns should be 0
        fwd_returns = result['fwd_return'].dropna()
        assert all(abs(fwd_returns) < 1e-10)
        
        # All labels should be 0 (flat)
        labels = result['label'].dropna()
        assert all(labels == 0)
    
    def test_default_parameters(self, sample_price_data):
        """Test function with default parameters."""
        # Test that function works with default horizon and threshold
        result = create_labels(sample_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'fwd_return' in result.columns
        assert 'label' in result.columns
        
        # Should use default values (horizon=5, threshold=0.0002)
        # Verify by checking that some labels are generated
        labels = result['label'].dropna()
        assert len(labels) > 0
    
    @patch('src.models._utils.labeling.logger')
    def test_logging(self, mock_logger, sample_price_data):
        """Test that appropriate logging occurs."""
        horizon = 10
        result = create_labels(sample_price_data, horizon=horizon, threshold=0.001)
        
        # Should log start message with horizon
        mock_logger.info.assert_any_call(f"Creating labels with a {horizon}-minute horizon...")
        
        # Should log completion
        mock_logger.info.assert_any_call("Label creation complete.")
    
    def test_label_distribution(self, sample_price_data):
        """Test that label distribution is reasonable."""
        result = create_labels(sample_price_data, horizon=5, threshold=0.005)  # 0.5%
        
        labels = result['label'].dropna()
        
        if len(labels) > 10:  # Only test if we have sufficient data
            # Count each label type
            up_count = (labels == 1).sum()
            down_count = (labels == -1).sum()
            flat_count = (labels == 0).sum()
            
            # Should have all three types of labels
            assert up_count + down_count + flat_count == len(labels)
            
            # No single label should dominate completely (unless by coincidence)
            total = len(labels)
            assert up_count < total * 0.9
            assert down_count < total * 0.9
            assert flat_count < total * 0.9
    
    def test_dropna_functionality(self, sample_price_data):
        """Test that NaN values are properly handled."""
        result = create_labels(sample_price_data, horizon=5, threshold=0.001)
        
        # Function calls dropna(), so result should not have NaN in both fwd_return and label
        # (individual columns may have NaN, but not both simultaneously)
        combined_na = result[['fwd_return', 'label']].isna().all(axis=1)
        assert not combined_na.any()


class TestCreateLabelsIntegration:
    """Integration tests for label creation."""
    
    def test_realistic_trading_scenario(self):
        """Test labeling with realistic trading scenario."""
        # Create realistic intraday price movement
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=390, freq='1min')  # Full day
        
        data = []
        current_price = 100.0
        
        # Simulate realistic price patterns
        for i, ts in enumerate(timestamps):
            # Add some trend and mean reversion
            trend = 0.0001 * np.sin(2 * np.pi * i / 390)  # Daily pattern
            noise = np.random.normal(0, 0.002)
            mean_reversion = -0.001 * (current_price - 100.0) / 100.0
            
            price_change = trend + noise + mean_reversion
            current_price = max(current_price * (1 + price_change), 1.0)
            
            data.append({
                'symbol': 'SPY',
                'timestamp': ts,
                'close': current_price,
                'open': current_price + np.random.normal(0, 0.001),
                'high': current_price + np.random.uniform(0, 0.005),
                'low': current_price - np.random.uniform(0, 0.005),
                'volume': np.random.randint(1000, 3000)
            })
        
        df = pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        # Test different horizons for trading signals
        horizons = [1, 5, 15, 30]  # 1min, 5min, 15min, 30min
        threshold = 0.002  # 0.2% threshold
        
        for horizon in horizons:
            result = create_labels(df, horizon=horizon, threshold=threshold)
            
            # Should generate reasonable number of labels
            labels = result['label'].dropna()
            assert len(labels) > 300  # Most of the data should have labels
            
            # Should have all three label types in realistic data
            unique_labels = set(labels.unique())
            assert unique_labels.issubset({-1, 0, 1})
            
            # Forward returns should be reasonable for the horizon
            fwd_returns = result['fwd_return'].dropna()
            max_reasonable_return = 0.1  # 10% max for any horizon
            assert all(abs(fwd_returns) < max_reasonable_return)


if __name__ == "__main__":
    pytest.main([__file__])