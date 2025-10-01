import pandas as pd
import numpy as np
import json
import os
import bisect
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

from v2.config import BACKTEST, STRATEGY, LSTM_MODEL
from v2.utils import get_data, preprocess_data, logger, log_params
from v2.model import create_StockPriceLSTM, predict_multiple_steps
from v2.strategy import get_signal, Signal


@dataclass
class BacktestOrder:
    """
    Simplified order for backtesting with take profit and stop loss levels.
    """
    order_id: str
    entry_price: float
    shares: float
    tp: float  # take profit price
    sl: float  # stop loss price
    timestamp: datetime  # when the position was opened

    def __lt__(self, other):
        return self.entry_price < other.entry_price


class BacktestOrderTracker:
    """
    Simplified order tracker for backtesting, adapted from v2/trader.py OrderTracker.
    Manages take profit and stop loss levels for open positions.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_price = 0.0
        self.orders: List[BacktestOrder] = []

    def update_price(self, price: float):
        """Update the current market price for threshold checking."""
        self.current_price = price

    def add_position(self, order_id: str, shares: float, entry_price: float, timestamp: datetime):
        """
        Add a new position with calculated take profit and stop loss levels.

        Args:
            order_id: Unique identifier for the order
            shares: Number of shares in the position
            entry_price: Price at which the position was entered
            timestamp: When the position was opened
        """
        order = BacktestOrder(
            order_id=order_id,
            entry_price=entry_price,
            shares=shares,
            tp=entry_price * STRATEGY['take_profit'],  # 20% profit target
            sl=entry_price * STRATEGY['stop_loss'],    # 20% stop loss
            timestamp=timestamp
        )
        bisect.insort(self.orders, order)
        logger.debug(f"Added position {order_id}: {shares:.2f} shares @ ${entry_price:.2f}, TP: ${order.tp:.2f}, SL: ${order.sl:.2f}")

    def check_exits(self) -> List[BacktestOrder]:
        """
        Check which orders should be closed due to TP/SL thresholds.

        Returns:
            List of orders that hit their TP or SL levels
        """
        triggered_orders = []
        for order in self.orders:
            if self.current_price >= order.tp or self.current_price <= order.sl:
                triggered_orders.append(order)
        return triggered_orders

    def execute_exits(self) -> List[BacktestOrder]:
        """
        Remove triggered orders and return them for execution.

        Returns:
            List of orders that were removed (to be sold)
        """
        triggered_orders = self.check_exits()

        # Remove triggered orders from tracking
        for order in triggered_orders:
            self.orders.remove(order)

        return triggered_orders

    def get_total_shares(self) -> float:
        """Get total shares across all open positions."""
        return sum(order.shares for order in self.orders)

    def has_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self.orders) > 0


def load_backtest_data() -> pd.DataFrame:
    """
    Load historical data for all symbols from BACKTEST config.

    Returns:
        Combined DataFrame with all symbol data sorted by timestamp
    """
    logger.info(f"Loading backtest data for {BACKTEST['symbols']}")
    logger.info(f"Date range: {BACKTEST['start_date']} to {BACKTEST['end_date']}")

    all_data = []
    for symbol in BACKTEST['symbols']:
        logger.info(f"Loading data for {symbol}")
        raw_data = get_data(symbol, BACKTEST['start_date'], BACKTEST['end_date'])
        processed_data = preprocess_data(raw_data)

        # Add symbol column for tracking
        processed_data['symbol'] = symbol
        all_data.append(processed_data)

    # Combine all data and sort by timestamp
    combined_data = pd.concat(all_data, axis=0).sort_index()
    logger.info(f"Loaded {len(combined_data)} total data points")

    return combined_data


def simulate_trading(data: pd.DataFrame, model) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run strategy on historical data and simulate trades.

    Args:
        data: Combined historical data for all symbols
        model: Trained LSTM model

    Returns:
        Tuple of (trades, portfolio_history, predictions_data)
    """
    logger.info("Starting trading simulation")

    trades = []
    portfolio_history = []
    predictions_data = []
    cash = BACKTEST['initial_capital']
    positions = {}  # symbol -> shares (for compatibility, but trackers now manage positions)
    input_length = LSTM_MODEL['input_length']

    # Initialize order trackers for each symbol
    order_trackers = {}
    for symbol in BACKTEST['symbols']:
        order_trackers[symbol] = BacktestOrderTracker(symbol)

    # Group data by symbol for processing
    symbols_data = {}
    for symbol in BACKTEST['symbols']:
        symbol_mask = data['symbol'] == symbol
        symbol_data = data[symbol_mask].drop('symbol', axis=1)
        symbols_data[symbol] = symbol_data

    # Get all unique timestamps and process chronologically
    # Extract just the timestamp level from MultiIndex (level 1, since level 0 is symbol)
    all_timestamps = sorted(data.index.get_level_values(1).unique())

    for i, timestamp in enumerate(all_timestamps):
        # Skip early timestamps where we don't have enough history
        if i < input_length:
            continue

        # Calculate current portfolio value for this timestamp
        total_position_value = 0
        current_prices = {}

        for symbol in BACKTEST['symbols']:
            symbol_data = symbols_data[symbol]
            if timestamp in symbol_data.index:
                current_log_price = symbol_data.loc[timestamp].iloc[0]  # log_close is first feature
                current_price = np.exp(current_log_price)
                current_prices[symbol] = current_price

                # Use order tracker to get accurate position value
                total_shares = order_trackers[symbol].get_total_shares()
                total_position_value += total_shares * current_price

        # Track portfolio history
        portfolio_snapshot = {
            'timestamp': timestamp,
            'cash': cash,
            'position_value': total_position_value,
            'total_value': cash + total_position_value,
            'daily_return': 0.0  # Will calculate below
        }

        # Calculate daily return if we have previous data
        if portfolio_history:
            prev_value = portfolio_history[-1]['total_value']
            if prev_value > 0:
                portfolio_snapshot['daily_return'] = (portfolio_snapshot['total_value'] - prev_value) / prev_value

        portfolio_history.append(portfolio_snapshot)

        # Process each symbol at this timestamp
        for symbol in BACKTEST['symbols']:
            symbol_data = symbols_data[symbol]

            # Check if we have data for this symbol at this timestamp
            if timestamp not in symbol_data.index:
                continue

            # Get the sequence for prediction (last input_length rows up to current timestamp)
            available_data = symbol_data[symbol_data.index <= timestamp]
            if len(available_data) < input_length:
                continue

            sequence = available_data.iloc[-input_length:].values

            try:
                # Get current price (log_close is first feature)
                current_log_price = sequence[-1, 0]  # Last row, first column
                current_price = np.exp(current_log_price)

                # Update price for TP/SL checking
                order_trackers[symbol].update_price(current_price)

                # Check for TP/SL exits BEFORE processing new signals
                exit_orders = order_trackers[symbol].execute_exits()
                for exit_order in exit_orders:
                    # Apply slippage to TP/SL exit price
                    if current_price >= exit_order.tp:
                        # Take profit - sell at lower price due to slippage
                        execution_price = current_price * (1 - BACKTEST['slippage'])
                        exit_reason = 'TAKE_PROFIT'
                    else:
                        # Stop loss - sell at lower price due to slippage
                        execution_price = current_price * (1 - BACKTEST['slippage'])
                        exit_reason = 'STOP_LOSS'

                    proceeds = exit_order.shares * execution_price - BACKTEST['commission']
                    cash += proceeds

                    # Update positions tracking for compatibility
                    positions[symbol] = positions.get(symbol, 0) - exit_order.shares

                    trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': exit_reason,
                        'shares': exit_order.shares,
                        'price': execution_price,
                        'value': exit_order.shares * execution_price,
                        'commission': BACKTEST['commission'],
                        'cash_after': cash,
                        'entry_price': exit_order.entry_price,
                        'entry_timestamp': exit_order.timestamp,
                        'pnl': (execution_price - exit_order.entry_price) * exit_order.shares - BACKTEST['commission']
                    })
                    logger.debug(f"{exit_reason} {exit_order.shares:.2f} shares of {symbol} at ${execution_price:.2f} (entry: ${exit_order.entry_price:.2f})")

                # Generate prediction and signal for new positions
                prediction = predict_multiple_steps(model, sequence)
                signal = get_signal(prediction)

                # Store prediction data for analysis
                predictions_data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'prediction_raw': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                    'prediction_cumsum': np.cumsum(prediction.flatten()).tolist() if isinstance(prediction, np.ndarray) else [prediction],
                    'signal': signal.name,
                    'current_price': current_price,
                    'current_log_price': current_log_price
                })

                # Execute trades based on signal
                if signal == Signal.BUY and cash > 0:
                    # Apply slippage to execution price (buy at higher price)
                    execution_price = current_price * (1 + BACKTEST['slippage'])

                    # Calculate position size
                    position_value = cash * STRATEGY['investment_size']
                    shares = position_value / execution_price
                    cost = shares * execution_price + BACKTEST['commission']

                    # if cost <= cash and cost > 100: # limit buys to positions > $100
                    if cost <= cash:
                        cash -= cost
                        positions[symbol] = positions.get(symbol, 0) + shares

                        # Add position to order tracker for TP/SL management
                        order_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(trades)}"
                        order_trackers[symbol].add_position(order_id, shares, execution_price, timestamp)

                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': execution_price,
                            'value': shares * execution_price,
                            'commission': BACKTEST['commission'],
                            'cash_after': cash
                        })
                        logger.debug(f"BUY {shares:.2f} shares of {symbol} at ${execution_price:.2f} (slippage: {BACKTEST['slippage']*100:.3f}%)")

                elif signal == Signal.SELL and order_trackers[symbol].has_positions():
                    # Apply slippage to execution price (sell at lower price)
                    execution_price = current_price * (1 - BACKTEST['slippage'])

                    # Sell all tracked positions (signal-based exit)
                    all_orders = order_trackers[symbol].orders.copy()  # Copy to avoid modification during iteration
                    total_shares = 0
                    total_pnl = 0

                    for order in all_orders:
                        proceeds = order.shares * execution_price - (BACKTEST['commission'] / len(all_orders))  # Split commission
                        cash += proceeds
                        total_shares += order.shares

                        # Calculate PnL for this specific order
                        order_pnl = (execution_price - order.entry_price) * order.shares - (BACKTEST['commission'] / len(all_orders))
                        total_pnl += order_pnl

                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': order.shares,
                            'price': execution_price,
                            'value': order.shares * execution_price,
                            'commission': BACKTEST['commission'] / len(all_orders),
                            'cash_after': cash,
                            'entry_price': order.entry_price,
                            'entry_timestamp': order.timestamp,
                            'pnl': order_pnl
                        })

                    # Clear all positions from tracker
                    order_trackers[symbol].orders.clear()
                    positions[symbol] = 0

                    logger.debug(f"SELL {total_shares:.2f} shares of {symbol} at ${execution_price:.2f} (signal-based, PnL: ${total_pnl:.2f})")

            except Exception as e:
                logger.error(f"Error processing {symbol} at {timestamp}: {e}")
                continue

    logger.info(f"Simulation complete. Executed {len(trades)} trades")
    return trades, portfolio_history, predictions_data


def calculate_metrics(trades: List[Dict], data: pd.DataFrame) -> Dict:
    """
    Calculate basic performance metrics from trades.

    Args:
        trades: List of trade records
        data: Historical data for final portfolio valuation

    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        logger.warning("No trades executed")
        return {
            'initial_capital': BACKTEST['initial_capital'],
            'final_cash': BACKTEST['initial_capital'],
            'position_value': 0.0,
            'final_portfolio_value': BACKTEST['initial_capital'],
            'total_return': 0.0,
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'avg_trade_value': 0.0,
        }

    # Calculate basic metrics
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    # Calculate final portfolio value
    final_cash = trades[-1]['cash_after'] if trades else BACKTEST['initial_capital']

    # Get final positions
    positions = {}
    for trade in trades:
        symbol = trade['symbol']
        if trade['action'] == 'BUY':
            positions[symbol] = positions.get(symbol, 0) + trade['shares']
        else:
            positions[symbol] = positions.get(symbol, 0) - trade['shares']

    # Value remaining positions at final prices
    position_value = 0
    for symbol, shares in positions.items():
        if shares > 0:
            # Get final price for this symbol
            symbol_data = data[data['symbol'] == symbol]
            if not symbol_data.empty:
                final_log_price = symbol_data.iloc[-1].iloc[0]  # First feature is log_close
                final_price = np.exp(final_log_price)
                position_value += shares * final_price

    final_portfolio_value = final_cash + position_value
    total_return = (final_portfolio_value - BACKTEST['initial_capital']) / BACKTEST['initial_capital'] * 100

    metrics = {
        'initial_capital': BACKTEST['initial_capital'],
        'final_cash': final_cash,
        'position_value': position_value,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'total_trades': len(trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'avg_trade_value': np.mean([t['value'] for t in trades]) if trades else 0,
    }

    return metrics


def save_backtest_results(trades: List[Dict], portfolio_history: List[Dict],
                         predictions_data: List[Dict], data: pd.DataFrame,
                         metrics: Dict) -> str:
    """
    Save complete backtest results to structured files for analysis.

    Args:
        trades: List of trade records
        portfolio_history: Daily portfolio snapshots
        predictions_data: Model predictions vs actual outcomes
        data: Combined historical market data
        metrics: Performance metrics

    Returns:
        results_dir: Path to created results directory
    """
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    results_dir = os.path.join("v2", "backtests", f"backtest_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Saving backtest results to {results_dir}")

    # 1. Save trades.csv
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(os.path.join(results_dir, "trades.csv"), index=False)
        logger.info(f"Saved {len(trades)} trades to trades.csv")

    # 2. Save portfolio_history.csv
    if portfolio_history:
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
        logger.info(f"Saved {len(portfolio_history)} portfolio snapshots to portfolio_history.csv")

    # 3. Save market_data.csv (raw OHLCV data used in backtest)
    # Extract original market data by removing 'symbol' column and reconstructing
    market_data = data.copy()
    market_data.to_csv(os.path.join(results_dir, "market_data.csv"))
    logger.info(f"Saved market data to market_data.csv")

    # 4. Save processed_features.csv (the 18 engineered features)
    processed_features = data.drop('symbol', axis=1, errors='ignore')
    processed_features.to_csv(os.path.join(results_dir, "processed_features.csv"))
    logger.info(f"Saved processed features to processed_features.csv")

    # 5. Save predictions_vs_actual.csv
    if predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(os.path.join(results_dir, "predictions_vs_actual.csv"), index=False)
        logger.info(f"Saved {len(predictions_data)} predictions to predictions_vs_actual.csv")

    # 6. Save metrics.json
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved performance metrics to metrics.json")

    # 7. Save config.json (complete configuration snapshot)
    config_snapshot = {
        'BACKTEST': BACKTEST,
        'STRATEGY': STRATEGY,
        'LSTM_MODEL': LSTM_MODEL,
        'backtest_metadata': {
            'run_timestamp': datetime.now().isoformat(),
            'symbols_analyzed': BACKTEST['symbols'],
            'total_trades': len(trades),
            'backtest_period_days': (BACKTEST['end_date'] - BACKTEST['start_date']).days
        }
    }

    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(config_snapshot, f, indent=2, default=str)
    logger.info("Saved complete configuration to config.json")

    logger.info(f"Backtest results successfully saved to {results_dir}")
    return results_dir


def print_results(metrics: Dict, trades: List[Dict]):
    """
    Print backtest results to console.

    Args:
        metrics: Performance metrics dictionary
        trades: List of trade records
    """
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\nPORTFOLIO PERFORMANCE:")
    print(f"Initial Capital:     ${metrics.get('initial_capital', BACKTEST['initial_capital']):,.2f}")
    print(f"Final Portfolio:     ${metrics.get('final_portfolio_value', 0):,.2f}")
    print(f"Total Return:        {metrics.get('total_return', 0):+.2f}%")
    print(f"Final Cash:          ${metrics.get('final_cash', 0):,.2f}")
    print(f"Position Value:      ${metrics.get('position_value', 0):,.2f}")

    print(f"\nTRADING ACTIVITY:")
    print(f"Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"Buy Trades:          {metrics.get('buy_trades', 0)}")
    print(f"Sell Trades:         {metrics.get('sell_trades', 0)}")
    print(f"Avg Trade Value:     ${metrics.get('avg_trade_value', 0):,.2f}")

    print(f"\nBACKTEST PARAMETERS:")
    print(f"Symbols:             {BACKTEST['symbols']}")
    print(f"Date Range:          {BACKTEST['start_date'].date()} to {BACKTEST['end_date'].date()}")
    print(f"Commission:          ${BACKTEST['commission']:.3f} per trade")
    print(f"Investment Size:     {STRATEGY['investment_size']*100:.1f}% per trade")

    if trades:
        print(f"\nRECENT TRADES:")
        recent_trades = trades[-5:] if len(trades) > 5 else trades
        for trade in recent_trades:
            action = trade['action']
            symbol = trade['symbol']
            shares = trade['shares']
            price = trade['price']
            date = trade['timestamp'].date() if hasattr(trade['timestamp'], 'date') else str(trade['timestamp'])[:10]
            print(f"  {date} | {action:4} | {symbol:4} | {shares:8.2f} shares @ ${price:7.2f}")

    print("="*60)


def main():
    """
    Main CLI entry point for backtesting.
    Uses configuration from v2/config.py, no arguments required.
    """
    logger.info("Starting backtest")

    try:
        # Load model
        logger.info("Loading trained model")
        model = create_StockPriceLSTM()

        # Load data
        data = load_backtest_data()

        # Run simulation
        trades, portfolio_history, predictions_data = simulate_trading(data, model)

        # Calculate metrics
        metrics = calculate_metrics(trades, data)

        # Print results
        print_results(metrics, trades)

        # Save complete results for analysis
        results_dir = save_backtest_results(trades, portfolio_history, predictions_data, data, metrics)
        print(f"\nDetailed results saved to: {results_dir}")

        logger.info("Backtest completed successfully")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == '__main__':
    main()