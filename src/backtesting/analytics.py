#!/usr/bin/env python3
"""
Backtesting Analytics and Optimal Return Comparison

Provides utilities to analyze trading performance and compare against optimal strategies.
Calculates buy-and-hold returns, perfect timing scenarios, and performance metrics.

Features:
- Optimal return calculations (buy-and-hold, perfect timing)
- Performance comparison and attribution
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and recovery periods
- Trade-level analysis and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats

from src.utils.logging_config import logger


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    excess_return_pct: float  # vs buy-and-hold

    # Risk metrics
    volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_duration_days: float
    recovery_time_days: float

    # Trade metrics
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_return_pct: float

    # Efficiency metrics
    optimal_return_capture_pct: float  # % of optimal return captured
    benchmark_outperformance_pct: float


@dataclass
class OptimalStrategy:
    """Optimal trading strategy results."""
    strategy_name: str
    total_return_pct: float
    daily_returns: List[float]
    portfolio_values: List[float]
    trade_signals: List[Dict]  # When to buy/sell for optimal results


class BacktestAnalyzer:
    """
    Comprehensive backtesting analysis and comparison tools.

    Compares actual trading results against optimal strategies and benchmarks.
    """

    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Initialize analyzer with historical price data.

        Args:
            historical_data: Dict of symbol -> DataFrame with OHLCV data
        """
        self.historical_data = historical_data
        self.symbols = list(historical_data.keys())

        # Calculate optimal strategies
        self.optimal_strategies = self._calculate_optimal_strategies()

        logger.info(f"BacktestAnalyzer initialized for {len(self.symbols)} symbols")

    def analyze_performance(self,
                          broker_results: Dict,
                          trade_history: pd.DataFrame,
                          daily_portfolio_values: List[Dict]) -> PerformanceMetrics:
        """
        Analyze trading performance and compare to optimal strategies.

        Args:
            broker_results: SimulatedBroker performance summary
            trade_history: DataFrame of all trades
            daily_portfolio_values: List of daily portfolio value records

        Returns:
            Comprehensive performance metrics
        """
        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(daily_portfolio_values)

        # Get benchmark (buy-and-hold) performance
        benchmark_return = self.optimal_strategies['buy_and_hold'].total_return_pct

        # Calculate risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)

        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(daily_portfolio_values)

        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics(trade_history)

        # Calculate optimal return capture
        optimal_return = self.optimal_strategies['perfect_timing'].total_return_pct
        optimal_capture = (broker_results['total_return_pct'] / optimal_return * 100) if optimal_return != 0 else 0

        # Annualized return
        days = len(daily_portfolio_values)
        if days > 0:
            annualized_return = ((1 + broker_results['total_return_pct'] / 100) ** (252 / days) - 1) * 100
        else:
            annualized_return = 0

        return PerformanceMetrics(
            total_return_pct=broker_results['total_return_pct'],
            annualized_return_pct=annualized_return,
            excess_return_pct=broker_results['total_return_pct'] - benchmark_return,
            volatility_pct=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=annualized_return / drawdown_metrics['max_drawdown'] if drawdown_metrics['max_drawdown'] > 0 else 0,
            max_drawdown_pct=drawdown_metrics['max_drawdown'],
            avg_drawdown_pct=drawdown_metrics['avg_drawdown'],
            drawdown_duration_days=drawdown_metrics['avg_duration'],
            recovery_time_days=drawdown_metrics['avg_recovery'],
            total_trades=broker_results['total_trades'],
            win_rate_pct=broker_results['win_rate'],
            profit_factor=broker_results['profit_factor'],
            avg_trade_return_pct=trade_metrics['avg_trade_return'],
            optimal_return_capture_pct=optimal_capture,
            benchmark_outperformance_pct=broker_results['total_return_pct'] - benchmark_return
        )

    def generate_comparison_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            metrics: Performance metrics from analyze_performance

        Returns:
            Formatted report string
        """
        optimal_returns = {name: strat.total_return_pct for name, strat in self.optimal_strategies.items()}

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTESTING PERFORMANCE REPORT            ║
╠══════════════════════════════════════════════════════════════╣

RETURN ANALYSIS
├─ Strategy Return:          {metrics.total_return_pct:>8.2f}%
├─ Annualized Return:        {metrics.annualized_return_pct:>8.2f}%
├─ Buy & Hold Return:        {optimal_returns['buy_and_hold']:>8.2f}%
├─ Perfect Timing Return:    {optimal_returns['perfect_timing']:>8.2f}%
├─ Excess Return vs B&H:     {metrics.excess_return_pct:>8.2f}%
└─ Optimal Return Capture:   {metrics.optimal_return_capture_pct:>8.2f}%

RISK METRICS
├─ Volatility (Annual):      {metrics.volatility_pct:>8.2f}%
├─ Sharpe Ratio:             {metrics.sharpe_ratio:>8.2f}
├─ Sortino Ratio:            {metrics.sortino_ratio:>8.2f}
├─ Calmar Ratio:             {metrics.calmar_ratio:>8.2f}
├─ Max Drawdown:             {metrics.max_drawdown_pct:>8.2f}%
├─ Avg Drawdown:             {metrics.avg_drawdown_pct:>8.2f}%
├─ Drawdown Duration:        {metrics.drawdown_duration_days:>8.1f} days
└─ Recovery Time:            {metrics.recovery_time_days:>8.1f} days

TRADING ACTIVITY
├─ Total Trades:             {metrics.total_trades:>8d}
├─ Win Rate:                 {metrics.win_rate_pct:>8.2f}%
├─ Profit Factor:            {metrics.profit_factor:>8.2f}
└─ Avg Trade Return:         {metrics.avg_trade_return_pct:>8.2f}%

STRATEGY EVALUATION
├─ Outperformed B&H:         {"✓" if metrics.excess_return_pct > 0 else "✗"}
├─ Risk-Adjusted Return:     {"✓" if metrics.sharpe_ratio > 1.0 else "✗"}
├─ Drawdown Control:         {"✓" if metrics.max_drawdown_pct < 15 else "✗"}
└─ Overall Grade:            {self._calculate_strategy_grade(metrics)}

╚══════════════════════════════════════════════════════════════╝
        """.strip()

        return report
        

    def _calculate_optimal_strategies(self) -> Dict[str, OptimalStrategy]:
        """Calculate optimal trading strategies for comparison."""
        strategies = {}

        # Buy and Hold strategy
        strategies['buy_and_hold'] = self._calculate_buy_and_hold()

        # Perfect timing strategy
        strategies['perfect_timing'] = self._calculate_perfect_timing()

        # Equal weight rebalancing
        strategies['equal_weight'] = self._calculate_equal_weight_rebalancing()

        return strategies

    def _calculate_buy_and_hold(self) -> OptimalStrategy:
        """Calculate buy-and-hold returns for equal-weighted portfolio."""
        if not self.historical_data:
            return OptimalStrategy("Buy & Hold", 0.0, [], [], [])

        # Calculate equal-weighted portfolio returns
        all_returns = []
        for symbol, df in self.historical_data.items():
            if len(df) > 1:
                first_price = df.iloc[0]['close']
                last_price = df.iloc[-1]['close']
                symbol_return = (last_price - first_price) / first_price
                all_returns.append(symbol_return)

        if not all_returns:
            return OptimalStrategy("Buy & Hold", 0.0, [], [], [])

        # Equal weighted return
        portfolio_return = np.mean(all_returns) * 100

        return OptimalStrategy(
            strategy_name="Buy & Hold",
            total_return_pct=portfolio_return,
            daily_returns=[],
            portfolio_values=[],
            trade_signals=[{'action': 'buy_all', 'timestamp': 'start'}]
        )

    def _calculate_perfect_timing(self) -> OptimalStrategy:
        """Calculate perfect timing strategy (always buy at lows, sell at highs)."""
        if not self.historical_data:
            return OptimalStrategy("Perfect Timing", 0.0, [], [], [])

        # For each symbol, find the maximum possible return
        total_return = 0
        trade_signals = []

        for symbol, df in self.historical_data.items():
            if len(df) < 2:
                continue

            # Find global min and max for each symbol
            min_price = df['close'].min()
            max_price = df['close'].max()

            # Calculate maximum possible return for this symbol
            symbol_max_return = (max_price - min_price) / min_price
            total_return += symbol_max_return

            # Record optimal trade signals
            min_idx = df['close'].idxmin()
            max_idx = df['close'].idxmax()

            if min_idx < max_idx:  # Buy then sell
                trade_signals.extend([
                    {'action': 'buy', 'symbol': symbol, 'price': min_price, 'timestamp': df.loc[min_idx, 'timestamp']},
                    {'action': 'sell', 'symbol': symbol, 'price': max_price, 'timestamp': df.loc[max_idx, 'timestamp']}
                ])

        # Average across symbols (equal weighting)
        avg_return = (total_return / len(self.historical_data)) * 100 if self.historical_data else 0

        return OptimalStrategy(
            strategy_name="Perfect Timing",
            total_return_pct=avg_return,
            daily_returns=[],
            portfolio_values=[],
            trade_signals=trade_signals
        )

    def _calculate_equal_weight_rebalancing(self) -> OptimalStrategy:
        """Calculate equal-weight rebalancing strategy (monthly rebalancing)."""
        # Simplified version - just return buy and hold for now
        return self._calculate_buy_and_hold()

    def _calculate_daily_returns(self, daily_values: List[Dict]) -> List[float]:
        """Calculate daily returns from portfolio values."""
        if len(daily_values) < 2:
            return []

        values = [d['portfolio_value'] for d in daily_values]
        returns = []

        for i in range(1, len(values)):
            if values[i-1] != 0:
                daily_return = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_return)

        return returns

    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not daily_returns or np.std(daily_returns) == 0:
            return 0.0

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)

        # Annualized Sharpe ratio
        excess_return = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = (excess_return / std_return) * np.sqrt(252)

        return sharpe

    def _calculate_sortino_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (Sharpe using only downside deviation)."""
        if not daily_returns:
            return 0.0

        mean_return = np.mean(daily_returns)
        downside_returns = [r for r in daily_returns if r < 0]

        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        excess_return = mean_return - (risk_free_rate / 252)
        sortino = (excess_return / downside_std) * np.sqrt(252)

        return sortino

    def _calculate_drawdown_metrics(self, daily_values: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(daily_values) < 2:
            return {'max_drawdown': 0, 'avg_drawdown': 0, 'avg_duration': 0, 'avg_recovery': 0}

        values = [d['portfolio_value'] for d in daily_values]

        # Calculate running maximum and drawdowns
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max * 100

        # Max drawdown
        max_drawdown = abs(np.min(drawdowns))

        # Average drawdown
        negative_drawdowns = [d for d in drawdowns if d < 0]
        avg_drawdown = abs(np.mean(negative_drawdowns)) if negative_drawdowns else 0

        # Drawdown periods analysis
        in_drawdown = False
        drawdown_start = 0
        drawdown_durations = []
        recovery_times = []

        for i, dd in enumerate(drawdowns):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:  # End of drawdown
                in_drawdown = False
                duration = i - drawdown_start
                drawdown_durations.append(duration)
                # Recovery time is the same as duration for now
                recovery_times.append(duration)

        avg_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        avg_recovery = np.mean(recovery_times) if recovery_times else 0

        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_duration': avg_duration,
            'avg_recovery': avg_recovery
        }

    def _calculate_trade_metrics(self, trade_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-level performance metrics."""
        if trade_history.empty:
            return {'avg_trade_return': 0}

        # Calculate per-trade returns (simplified)
        total_value = trade_history['total_value'].sum()
        num_trades = len(trade_history)

        avg_trade_value = total_value / num_trades if num_trades > 0 else 0

        return {
            'avg_trade_return': avg_trade_value  # Simplified metric
        }

    def _get_normalized_returns(self, strategy_name: str, dates: List[datetime]) -> List[float]:
        """Get normalized returns for comparison chart."""
        strategy = self.optimal_strategies.get(strategy_name)
        if not strategy or not dates:
            return [0] * len(dates)

        # For buy-and-hold, create linear progression
        if strategy_name == 'buy_and_hold':
            total_return = strategy.total_return_pct
            return [total_return * (i / (len(dates) - 1)) for i in range(len(dates))]

        # For perfect timing, assume step function at optimal times
        if strategy_name == 'perfect_timing':
            returns = [0] * len(dates)
            # Assume returns accrue linearly over time (simplified)
            total_return = strategy.total_return_pct
            for i in range(len(dates)):
                returns[i] = total_return * (i / (len(dates) - 1))
            return returns

        return [0] * len(dates)

    def _calculate_strategy_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall strategy grade based on multiple metrics."""
        score = 0

        # Return component (30%)
        if metrics.excess_return_pct > 10:
            score += 30
        elif metrics.excess_return_pct > 5:
            score += 25
        elif metrics.excess_return_pct > 0:
            score += 20
        elif metrics.excess_return_pct > -5:
            score += 10

        # Risk component (25%)
        if metrics.sharpe_ratio > 2.0:
            score += 25
        elif metrics.sharpe_ratio > 1.5:
            score += 20
        elif metrics.sharpe_ratio > 1.0:
            score += 15
        elif metrics.sharpe_ratio > 0.5:
            score += 10

        # Drawdown component (25%)
        if metrics.max_drawdown_pct < 5:
            score += 25
        elif metrics.max_drawdown_pct < 10:
            score += 20
        elif metrics.max_drawdown_pct < 15:
            score += 15
        elif metrics.max_drawdown_pct < 25:
            score += 10

        # Win rate component (20%)
        if metrics.win_rate_pct > 70:
            score += 20
        elif metrics.win_rate_pct > 60:
            score += 15
        elif metrics.win_rate_pct > 50:
            score += 10
        elif metrics.win_rate_pct > 40:
            score += 5

        # Convert to letter grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"


def run_backtest_analysis(historical_data: Dict[str, pd.DataFrame],
                         broker_results: Dict,
                         trade_history: pd.DataFrame,
                         daily_portfolio_values: List[Dict]) -> Tuple[PerformanceMetrics, str]:
    """
    Convenience function to run complete backtest analysis.

    Args:
        historical_data: Historical price data by symbol
        broker_results: SimulatedBroker performance summary
        trade_history: DataFrame of all trades
        daily_portfolio_values: List of daily portfolio value records

    Returns:
        Tuple of (performance metrics, comparison report)
    """
    analyzer = BacktestAnalyzer(historical_data)
    metrics = analyzer.analyze_performance(broker_results, trade_history, daily_portfolio_values)
    report = analyzer.generate_comparison_report(metrics)

    return metrics, report


if __name__ == "__main__":
    """Test the analytics module."""

    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = {
        'AAPL': pd.DataFrame({
            'timestamp': dates,
            'close': 150 + np.cumsum(np.random.randn(100) * 2)
        }),
        'MSFT': pd.DataFrame({
            'timestamp': dates,
            'close': 300 + np.cumsum(np.random.randn(100) * 3)
        })
    }

    # Create sample results
    broker_results = {
        'total_return_pct': 15.5,
        'total_trades': 25,
        'win_rate': 65.0,
        'profit_factor': 1.8
    }

    daily_values = [
        {'timestamp': date, 'portfolio_value': 100000 + i * 150}
        for i, date in enumerate(dates)
    ]

    trade_history = pd.DataFrame({
        'timestamp': dates[:10],
        'symbol': ['AAPL'] * 10,
        'total_value': [1000] * 10
    })

    # Run analysis
    metrics, report = run_backtest_analysis(sample_data, broker_results, trade_history, daily_values)

    print(report)
    print(f"\nOptimal Return Capture: {metrics.optimal_return_capture_pct:.1f}%")