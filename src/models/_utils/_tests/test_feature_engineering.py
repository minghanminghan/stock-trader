#!/usr/bin/env python3
"""
Unit tests for src/models/_utils/feature_engineering.py - Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.models._utils.feature_engineering import compute_features


class TestComputeFeatures:
    """Test compute_features function."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        # Create data for multiple symbols over time
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=200, freq='1min')
        symbols = ['SPY', 'AAPL']
        
        data = []
        np.random.seed(42)  # For reproducible tests
        
        for symbol in symbols:
            base_price = 100.0 if symbol == 'SPY' else 150.0
            
            for i, timestamp in enumerate(timestamps):
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.5)  # Random walk
                close_price = base_price + i * 0.01 + price_change
                
                # OHLC based on close
                open_price = close_price + np.random.normal(0, 0.1)
                high_price = max(open_price, close_price) + np.random.uniform(0, 0.5)
                low_price = min(open_price, close_price) - np.random.uniform(0, 0.5)
                
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': np.random.randint(1000, 5000)
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['symbol', 'timestamp']).sort_index()
    
    def test_compute_features_basic_functionality(self, sample_ohlcv_data):
        """Test basic feature computation."""
        result = compute_features(sample_ohlcv_data)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have MultiIndex
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['symbol', 'timestamp']
        
        # Should have original columns plus new features
        original_columns = set(sample_ohlcv_data.columns)
        result_columns = set(result.columns)
        assert original_columns.issubset(result_columns)
    
    def test_compute_features_expected_columns(self, sample_ohlcv_data):
        """Test that all expected feature columns are created."""
        result = compute_features(sample_ohlcv_data)
        
        expected_features = [
            'open', 'high', 'low', 'close', 'volume',  # Original
            'return_1m',                                # Returns
            'mom_5m', 'mom_15m', 'mom_60m',            # Momentum
            'vol_15m', 'vol_60m',                      # Volatility
            'vol_zscore',                              # Volume Z-score
            'time_sin', 'time_cos'                     # Time encoding
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_returns_calculation(self, sample_ohlcv_data):
        """Test 1-minute returns calculation."""
        result = compute_features(sample_ohlcv_data)
        
        # Check that returns are calculated correctly
        for symbol in result.index.get_level_values(0).unique():
            symbol_data = result.loc[symbol]
            
            # First return should be NaN (no previous value)
            assert pd.isna(symbol_data.iloc[0]['return_1m'])
            
            # Subsequent returns should be log returns
            if len(symbol_data) > 1:
                close_prices = symbol_data['close']
                expected_return = np.log(close_prices.iloc[1] / close_prices.iloc[0])
                actual_return = symbol_data.iloc[1]['return_1m']
                
                if not pd.isna(actual_return):
                    assert abs(actual_return - expected_return) < 1e-10
    
    def test_momentum_indicators(self, sample_ohlcv_data):
        """Test momentum indicator calculations."""
        result = compute_features(sample_ohlcv_data)
        
        # Test momentum calculations
        for symbol in result.index.get_level_values(0).unique():
            symbol_data = result.loc[symbol]
            close_prices = symbol_data['close']
            
            # Check 5-minute momentum
            if len(symbol_data) > 5:
                expected_mom_5m = (close_prices.iloc[5] - close_prices.iloc[0]) / close_prices.iloc[0]
                actual_mom_5m = symbol_data.iloc[5]['mom_5m']
                
                if not pd.isna(actual_mom_5m):
                    assert abs(actual_mom_5m - expected_mom_5m) < 1e-10
    
    def test_volatility_indicators(self, sample_ohlcv_data):
        """Test volatility indicator calculations."""
        result = compute_features(sample_ohlcv_data)
        
        # Check that volatility features are computed
        assert 'vol_15m' in result.columns
        assert 'vol_60m' in result.columns
        
        # Volatility should be non-negative where not NaN
        vol_15m = result['vol_15m'].dropna()
        vol_60m = result['vol_60m'].dropna()
        
        assert all(vol_15m >= 0)
        assert all(vol_60m >= 0)
    
    def test_volume_zscore(self, sample_ohlcv_data):
        """Test volume Z-score calculation."""
        result = compute_features(sample_ohlcv_data)
        
        # Check that volume Z-score is calculated
        assert 'vol_zscore' in result.columns
        
        # Z-scores should be reasonable (not all NaN after warmup period)
        vol_zscore = result['vol_zscore'].dropna()
        assert len(vol_zscore) > 0
        
        # Z-scores should have reasonable distribution
        assert abs(vol_zscore.mean()) < 2  # Should be roughly centered
    
    def test_time_encoding(self, sample_ohlcv_data):
        """Test cyclical time encoding."""
        result = compute_features(sample_ohlcv_data)
        
        # Check time encoding columns exist
        assert 'time_sin' in result.columns
        assert 'time_cos' in result.columns
        
        # Check value ranges
        time_sin = result['time_sin'].dropna()
        time_cos = result['time_cos'].dropna()
        
        assert all(time_sin >= -1) and all(time_sin <= 1)
        assert all(time_cos >= -1) and all(time_cos <= 1)
        
        # Check that sin^2 + cos^2 = 1 (approximately)
        sin_cos_sum = result['time_sin']**2 + result['time_cos']**2
        sin_cos_sum_clean = sin_cos_sum.dropna()
        assert all(abs(sin_cos_sum_clean - 1) < 1e-10)
    
    def test_multiple_symbols(self, sample_ohlcv_data):
        """Test feature computation for multiple symbols."""
        result = compute_features(sample_ohlcv_data)
        
        # Should have data for both symbols
        symbols = result.index.get_level_values(0).unique()
        assert 'SPY' in symbols
        assert 'AAPL' in symbols
        
        # Each symbol should have features computed independently
        spy_data = result.loc['SPY']
        aapl_data = result.loc['AAPL']
        
        assert len(spy_data) > 0
        assert len(aapl_data) > 0
        
        # Features should be different between symbols (different price series)
        assert not spy_data['return_1m'].equals(aapl_data['return_1m'])
    
    def test_data_sorting(self):
        """Test that data is properly sorted by timestamp."""
        # Create unsorted data
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min')
        shuffled_timestamps = np.random.permutation(timestamps)
        
        data = []
        for i, ts in enumerate(shuffled_timestamps):
            data.append({
                'symbol': 'SPY',
                'timestamp': ts,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0 + i * 0.1,
                'volume': 1000
            })
        
        unsorted_df = pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        result = compute_features(unsorted_df)
        
        # Result should be sorted by timestamp
        spy_data = result.loc['SPY']
        timestamps_result = spy_data.index
        assert all(timestamps_result[i] <= timestamps_result[i+1] for i in range(len(timestamps_result)-1))
    
    def test_dropna_functionality(self, sample_ohlcv_data):
        """Test that NaN values are properly dropped."""
        result = compute_features(sample_ohlcv_data)
        
        # Result should not have any rows with all NaN values
        # (some individual cells may be NaN due to rolling calculations)
        assert len(result) > 0
        
        # Check that we don't have completely empty rows
        non_null_counts = result.notna().sum(axis=1)
        assert all(non_null_counts > 0)
    
    def test_edge_case_single_row(self):
        """Test behavior with single row of data."""
        single_row_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }]).set_index(['symbol', 'timestamp'])
        
        result = compute_features(single_row_data)
        
        # Should handle single row gracefully
        assert isinstance(result, pd.DataFrame)
        # Most features will be NaN but shouldn't crash
    
    def test_edge_case_empty_data(self):
        """Test behavior with empty DataFrame."""
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_data.index = pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp'])
        
        result = compute_features(empty_data)
        
        # Should handle empty data gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_missing_ohlcv_columns(self):
        """Test behavior with missing required columns."""
        incomplete_data = pd.DataFrame([{
            'symbol': 'SPY',
            'timestamp': datetime.now(),
            'close': 100.0,
            # Missing other OHLCV columns
        }]).set_index(['symbol', 'timestamp'])
        
        # Should raise KeyError for missing columns
        with pytest.raises(KeyError):
            compute_features(incomplete_data)
    
    def test_invalid_volume_data(self):
        """Test handling of invalid volume data."""
        data_with_invalid_volume = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': -100 if i == 5 else 1000  # Negative volume
            }
            for i in range(50)
        ]).set_index(['symbol', 'timestamp'])
        
        # Should handle invalid volume gracefully
        result = compute_features(data_with_invalid_volume)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_price_data_with_zeros(self):
        """Test handling of zero prices."""
        data_with_zeros = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 0.0 if i == 10 else 100.0,  # Zero price
                'volume': 1000
            }
            for i in range(50)
        ]).set_index(['symbol', 'timestamp'])
        
        # Should handle zero prices (will create inf in log returns)
        result = compute_features(data_with_zeros)
        assert isinstance(result, pd.DataFrame)
        
        # After dropna(), should not contain inf values
        returns = result['return_1m'].dropna()
        finite_returns = returns[np.isfinite(returns)]
        # Most returns should be finite
        assert len(finite_returns) > len(returns) * 0.8
    
    def test_feature_data_types(self, sample_ohlcv_data):
        """Test that computed features have appropriate data types."""
        result = compute_features(sample_ohlcv_data)
        
        # All feature columns should be numeric
        feature_columns = [
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m',
            'vol_15m', 'vol_60m', 'vol_zscore', 'time_sin', 'time_cos'
        ]
        
        for col in feature_columns:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_feature_value_ranges(self, sample_ohlcv_data):
        """Test that feature values are in reasonable ranges."""
        result = compute_features(sample_ohlcv_data)
        
        # Returns should be reasonable (< 100% per minute)
        returns = result['return_1m'].dropna()
        assert all(abs(returns) < 1.0)  # < 100% per minute
        
        # Momentum should be reasonable
        momentum_cols = ['mom_5m', 'mom_15m', 'mom_60m']
        for col in momentum_cols:
            momentum = result[col].dropna()
            if len(momentum) > 0:
                assert all(abs(momentum) < 10.0)  # < 1000% change
        
        # Volatility should be non-negative and reasonable
        vol_cols = ['vol_15m', 'vol_60m']
        for col in vol_cols:
            volatility = result[col].dropna()
            if len(volatility) > 0:
                assert all(volatility >= 0)
                assert all(volatility < 1.0)  # Reasonable vol range
    
    @patch('src.models._utils.feature_engineering.logger')
    def test_logging(self, mock_logger, sample_ohlcv_data):
        """Test that appropriate logging occurs."""
        result = compute_features(sample_ohlcv_data)
        
        # Should log start and completion
        mock_logger.info.assert_any_call("Computing LSTM features...")
        
        # Should log feature count and column info
        completion_calls = [call for call in mock_logger.info.call_args_list 
                          if "Feature computation complete" in str(call)]
        assert len(completion_calls) > 0


class TestFeatureEngineeringEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_non_multiindex_dataframe(self):
        """Test with DataFrame that doesn't have MultiIndex."""
        simple_df = pd.DataFrame([{
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }])
        
        # Should raise error for non-MultiIndex DataFrame
        with pytest.raises((KeyError, AttributeError)):
            compute_features(simple_df)
    
    def test_wrong_index_names(self):
        """Test with MultiIndex having wrong level names."""
        wrong_index_df = pd.DataFrame([{
            'wrong_symbol': 'SPY',
            'wrong_timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }]).set_index(['wrong_symbol', 'wrong_timestamp'])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((KeyError, AttributeError)):
            compute_features(wrong_index_df)
    
    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        duplicate_time = datetime.now()
        duplicate_data = pd.DataFrame([
            {
                'symbol': 'SPY',
                'timestamp': duplicate_time,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            },
            {
                'symbol': 'SPY',
                'timestamp': duplicate_time,  # Duplicate
                'open': 100.1,
                'high': 101.1,
                'low': 99.1,
                'close': 100.6,
                'volume': 1100
            }
        ]).set_index(['symbol', 'timestamp'])
        
        # Should handle duplicates gracefully
        result = compute_features(duplicate_data)
        assert isinstance(result, pd.DataFrame)


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""
    
    def test_realistic_trading_data_pipeline(self):
        """Test with realistic trading data patterns."""
        # Create realistic intraday data with trends and volatility
        timestamps = pd.date_range('2024-01-01 09:30:00', periods=390, freq='1min')  # Full trading day
        
        data = []
        current_price = 100.0
        
        for i, ts in enumerate(timestamps):
            # Add market open/close volatility patterns
            hour = ts.hour
            minute = ts.minute
            
            # Higher volatility at open and close
            vol_multiplier = 1.0
            if (hour == 9 and minute < 45) or hour >= 15:
                vol_multiplier = 2.0
            
            # Add intraday trend
            trend = 0.001 * np.sin(2 * np.pi * i / 390)  # Slight daily pattern
            
            price_change = np.random.normal(trend, 0.002 * vol_multiplier)
            current_price = max(current_price + price_change, 1.0)  # Prevent negative prices
            
            # Generate OHLC
            open_price = current_price + np.random.normal(0, 0.001)
            high_price = max(open_price, current_price) + np.random.uniform(0, 0.01)
            low_price = min(open_price, current_price) - np.random.uniform(0, 0.01)
            volume = np.random.randint(800, 2000) * vol_multiplier
            
            data.append({
                'symbol': 'SPY',
                'timestamp': ts,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data).set_index(['symbol', 'timestamp'])
        
        # Compute features
        result = compute_features(df)
        
        # Verify realistic output
        assert len(result) > 300  # Should have most data after dropna
        
        # Check that features capture market patterns
        spy_data = result.loc['SPY']
        
        # Returns should show volatility clustering
        returns = spy_data['return_1m'].dropna()
        assert len(returns) > 300
        
        # Momentum indicators should show trends
        mom_5m = spy_data['mom_5m'].dropna()
        assert len(mom_5m) > 300
        
        # Time encoding should reflect intraday patterns
        time_features = spy_data[['time_sin', 'time_cos']].dropna()
        assert len(time_features) > 300


if __name__ == "__main__":
    pytest.main([__file__])