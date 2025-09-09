#!/usr/bin/env python3
"""
Backtest pre-trained model on historical data.
Tests saved models from src/models/weights on unseen time periods.

Usage:
    python backtest_model.py --start 2024-01-01 --end 2024-06-01
    python backtest_model.py --model src/models/weights/lgbm_model_20250908_170349.pkl --symbols AAPL MSFT
"""

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple

from src.models.prediction import ModelLoader, SignalGenerator
from src.data_ingestion import fetch_stock_data
from src.feature_engineering import compute_features
from src.labeling import create_labels
from src.config import DATA_DIR
from src.utils.logging_config import logger


class ModelBacktester:
    """
    Backtest pre-trained models on historical data.
    Loads model from weights and tests on specified date range.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize backtester.
        
        Args:
            model_path: Path to model file (uses latest if None)
        """
        self.model_loader = ModelLoader()
        self.signal_generator = SignalGenerator(model_path)
        self.model_path = model_path or self.model_loader.get_latest_model_path()
        
        logger.info(f"Backtester initialized with model: {os.path.basename(self.model_path) if self.model_path else 'None'}")
    
    def load_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data for backtesting.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Combined DataFrame with all symbols
        """
        logger.info(f"Loading historical data: {symbols} from {start_date} to {end_date}")
        
        # Fetch data if not already cached
        fetch_stock_data(tickers=symbols, start_date=start_date, end_date=end_date)
        
        # Load cached data
        all_data = []
        for symbol in symbols:
            file_path = os.path.join(DATA_DIR, f"{symbol}_1min_{start_date}_to_{end_date}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"Data file not found for {symbol}: {file_path}")
        
        if not all_data:
            raise ValueError("No historical data found")
        
        return pd.concat(all_data)
    
    def generate_signals_historical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for historical data using pre-trained model.
        
        Args:
            df: Historical OHLCV data with MultiIndex (symbol, timestamp)
            
        Returns:
            DataFrame with signals added
        """
        logger.info("Generating historical signals...")
        
        # Compute features
        featured_df = compute_features(df)
        
        if featured_df.empty:
            raise ValueError("Feature computation failed")
        
        # Expected features from training
        expected_features = [
            'return_1m', 'mom_5m', 'mom_15m', 'mom_60m', 
            'vol_15m', 'vol_60m', 'vol_zscore', 
            'time_sin', 'time_cos'
        ]
        
        # Load model
        model = self.model_loader.load_model(self.model_path)
        if model is None:
            raise ValueError(f"Could not load model from {self.model_path}")
        
        # Vectorized batch prediction (much faster than row-by-row)
        total_rows = len(featured_df)
        logger.info(f"Processing {total_rows:,} rows for predictions...")
        
        # Identify valid rows (all features present)
        valid_mask = featured_df[expected_features].notna().all(axis=1)
        valid_rows = featured_df[valid_mask]
        
        logger.info(f"Found {len(valid_rows):,} valid rows with all features")
        
        # Initialize arrays
        predictions = np.ones(total_rows, dtype=int)  # Default to HOLD (1)
        confidences = np.zeros(total_rows)
        prob_sell = np.full(total_rows, 0.33)
        prob_hold = np.full(total_rows, 0.34) 
        prob_buy = np.full(total_rows, 0.33)
        
        if len(valid_rows) > 0:
            # Batch prediction on all valid rows at once
            X_batch = valid_rows[expected_features].values
            
            logger.info("Running batch prediction...")
            raw_predictions = model.predict(X_batch)
            batch_probabilities = model.predict_proba(X_batch)
            batch_confidences = np.max(batch_probabilities, axis=1)
            
            # Map model predictions to standardized signals
            # Model outputs: {-1: DOWN, 0: FLAT, 1: UP}
            # Convert to: {0: SELL, 1: HOLD, 2: BUY}
            mapped_predictions = np.where(raw_predictions == -1, 0,  # DOWN -> SELL
                                 np.where(raw_predictions == 0, 1,   # FLAT -> HOLD
                                 np.where(raw_predictions == 1, 2,   # UP -> BUY
                                         1)))                        # Default to HOLD
            
            # Update arrays for valid indices
            predictions[valid_mask] = mapped_predictions
            confidences[valid_mask] = batch_confidences
            prob_sell[valid_mask] = batch_probabilities[:, 0]
            prob_hold[valid_mask] = batch_probabilities[:, 1] 
            prob_buy[valid_mask] = batch_probabilities[:, 2]
        
        logger.info("Batch prediction complete")
        
        # Add predictions to dataframe
        featured_df['ml_signal'] = predictions
        featured_df['ml_confidence'] = confidences
        featured_df['prob_sell'] = prob_sell
        featured_df['prob_hold'] = prob_hold
        featured_df['prob_buy'] = prob_buy
        
        logger.info(f"Generated {len(predictions)} predictions")
        return featured_df
    
    def simulate_trading(self, df: pd.DataFrame, 
                        initial_capital: float = 100000,
                        position_size: float = 0.02) -> Dict:
        """
        Simulate trading based on model signals.
        
        Args:
            df: DataFrame with signals
            initial_capital: Starting capital
            position_size: Fraction of capital per trade
            
        Returns:
            Trading simulation results
        """
        logger.info("Simulating trading strategy...")
        
        # Initialize tracking variables
        capital = initial_capital
        positions = {}  # symbol -> (qty, entry_price, entry_time)
        trades = []
        equity_curve = []
        
        # Group by symbol for processing
        for symbol in df.index.get_level_values(0).unique():
            symbol_data = df.loc[symbol].sort_index()
            
            for timestamp, row in symbol_data.iterrows():
                current_price = row['close']
                signal = row['ml_signal']
                confidence = row['ml_confidence']
                
                # Skip low confidence signals (lowered from 0.6 to 0.4)
                if confidence < 0.4:
                    continue
                
                # Signals are now standardized: {0: SELL, 1: HOLD, 2: BUY}
                if signal not in [0, 1, 2]:
                    continue  # Skip unexpected values
                
                # Current position
                current_pos = positions.get(symbol)
                
                # Trading logic
                if signal == 2 and current_pos is None:  # BUY signal, no position
                    # Enter long position
                    trade_value = capital * position_size
                    qty = int(trade_value / current_price)
                    if qty > 0:
                        positions[symbol] = (qty, current_price, timestamp)
                        capital -= qty * current_price
                        trades.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'qty': qty,
                            'price': current_price,
                            'value': qty * current_price,
                            'confidence': confidence
                        })
                
                elif signal == 0 and current_pos is not None:  # SELL signal, have position
                    # Exit long position
                    qty, entry_price, entry_time = current_pos
                    exit_value = qty * current_price
                    capital += exit_value
                    
                    pnl = exit_value - (qty * entry_price)
                    hold_time = (timestamp - entry_time).total_seconds() / 60  # minutes
                    
                    trades.append({
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'qty': qty,
                        'price': current_price,
                        'value': exit_value,
                        'confidence': confidence,
                        'entry_price': entry_price,
                        'pnl': pnl,
                        'hold_time_min': hold_time
                    })
                    
                    del positions[symbol]
                
                # Calculate current equity
                portfolio_value = sum(qty * current_price for qty, _, _ in positions.values())
                total_equity = capital + portfolio_value
                equity_curve.append({
                    'timestamp': timestamp,
                    'cash': capital,
                    'positions_value': portfolio_value,
                    'total_equity': total_equity
                })
        
        # Calculate final results
        final_equity = capital + sum(qty * df.loc[symbol].iloc[-1]['close'] 
                                   for symbol, (qty, _, _) in positions.items())
        
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Calculate trade statistics
        trade_pnls = [trade['pnl'] for trade in trades if 'pnl' in trade]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]
        
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len([t for t in trades if 'pnl' in t]),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trade_pnls) if trade_pnls else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else np.inf,
            'trades': trades,
            'equity_curve': equity_curve,
            'open_positions': positions
        }
        
        return results
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str,
                    initial_capital: float = 100000) -> Dict:
        """
        Run complete backtest.
        
        Args:
            symbols: Stock symbols to test
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest: {symbols} from {start_date} to {end_date}")
        
        # Load historical data
        historical_data = self.load_historical_data(symbols, start_date, end_date)
        
        # Generate signals
        signals_df = self.generate_signals_historical(historical_data)
        
        # Simulate trading
        results = self.simulate_trading(signals_df, initial_capital)
        
        # Add metadata
        results['symbols'] = symbols
        results['start_date'] = start_date
        results['end_date'] = end_date
        results['model_path'] = self.model_path
        
        return results


def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Model: {os.path.basename(results['model_path'])}")
    print(f"Symbols: {', '.join(results['symbols'])}")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Average Win: ${results['avg_win']:.2f}")
    print(f"  Average Loss: ${results['avg_loss']:.2f}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Open Positions: {len(results['open_positions'])}")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest pre-trained model")
    
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'AMZN', 'NVDA'],
                       help='Stock symbols to test')
    parser.add_argument('--model', help='Specific model file to use')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    
    args = parser.parse_args()
    
    try:
        # Initialize backtester
        backtester = ModelBacktester(args.model)
        
        # Run backtest
        results = backtester.run_backtest(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital
        )
        
        # Print results
        print_backtest_results(results)
        
        # Optional: Save detailed results
        # output_file = f"backtest_results_{args.start}_{args.end}.csv"
        # trades_df = pd.DataFrame(results['trades'])
        # if not trades_df.empty:
        #     trades_df.to_csv(output_file, index=False)
        #     print(f"\nDetailed trades saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())