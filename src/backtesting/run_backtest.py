#!/usr/bin/env python3
"""
Backtesting Runner Script

Simple script to run backtesting and generate performance comparison reports.

Usage:
    python run_backtest.py          # Run quick backtest
    python run_backtest.py --full   # Run full backtest with all symbols
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.main import run_backtest_comparison, OrchestratorConfig, TradingOrchestrator
from src.config import BACKTESTING_TICKERS, BACKTESTING_START_DATE, BACKTESTING_END_DATE
from src.utils.logging_config import logger


def run_quick_backtest():
    """Run a quick backtest with limited symbols."""
    logger.info("Starting quick backtest...")

    config = OrchestratorConfig(
        is_backtesting=True,
        backtest_symbols=["AAPL", "MSFT", "NVDA"],  # Just 3 symbols for speed
        backtest_start_date=BACKTESTING_START_DATE,
        backtest_end_date=BACKTESTING_END_DATE,
        initial_capital=100000.0,
        min_data_points=60,
    )

    orchestrator = TradingOrchestrator(config)

    try:
        orchestrator.start()
        return orchestrator.backtest_results
    except Exception as e:
        logger.error(f"Quick backtest failed: {e}")
        return None
    finally:
        orchestrator.stop()


def run_full_backtest():
    """Run a comprehensive backtest with all symbols."""
    logger.info("Starting full backtest...")

    config = OrchestratorConfig(
        is_backtesting=True,
        backtest_symbols=BACKTESTING_TICKERS,  # All symbols
        backtest_start_date=BACKTESTING_START_DATE,
        backtest_end_date=BACKTESTING_END_DATE,
        initial_capital=100000.0,
        min_data_points=60,
    )

    orchestrator = TradingOrchestrator(config)

    try:
        orchestrator.start()
        return orchestrator.backtest_results
    except Exception as e:
        logger.error(f"Full backtest failed: {e}")
        return None
    finally:
        orchestrator.stop()


def main():
    """Main entry point."""
    # Check for command line arguments
    full_mode = "--full" in sys.argv

    print("🤖 Stock Trading System - Backtesting Runner")
    print("=" * 50)

    if full_mode:
        print(f"Running full backtest with {len(BACKTESTING_TICKERS)} symbols...")
        results = run_full_backtest()
    else:
        print("Running quick backtest with 3 symbols...")
        print("💡 Use --full flag for comprehensive backtest")
        results = run_quick_backtest()

    if results:
        metrics = results['metrics']

        print("\n" + "=" * 50)
        print("Backtest completed")
        print("=" * 50)

        print(f"Total Return:     {metrics.total_return_pct:>8.2f}%")
        print(f"Sharpe Ratio:     {metrics.sharpe_ratio:>8.2f}")
        print(f"Max Drawdown:     {metrics.max_drawdown_pct:>8.2f}%")
        print(f"Win Rate:         {metrics.win_rate_pct:>8.2f}%")
        print(f"Total Trades:     {metrics.total_trades:>8d}")
        print(f"Strategy Grade:   {_calculate_grade(metrics):>8s}")

        print(f"\nOptimal Return Capture: {metrics.optimal_return_capture_pct:.1f}%")
        print(f"vs Buy & Hold: {'+' if metrics.excess_return_pct > 0 else ''}{metrics.excess_return_pct:.2f}%")

        print("\n" + "=" * 50)
        print("Results saved and ready for analysis!")

    else:
        print("\nBacktest failed. Check logs for details.")
        return 1

    return 0


def _calculate_grade(metrics):
    """Calculate a simple grade for the strategy."""
    score = 0

    # Return component
    if metrics.excess_return_pct > 10:
        score += 40
    elif metrics.excess_return_pct > 5:
        score += 30
    elif metrics.excess_return_pct > 0:
        score += 20

    # Risk component
    if metrics.sharpe_ratio > 2.0:
        score += 30
    elif metrics.sharpe_ratio > 1.5:
        score += 25
    elif metrics.sharpe_ratio > 1.0:
        score += 20
    elif metrics.sharpe_ratio > 0.5:
        score += 10

    # Drawdown component
    if metrics.max_drawdown_pct < 5:
        score += 30
    elif metrics.max_drawdown_pct < 10:
        score += 25
    elif metrics.max_drawdown_pct < 15:
        score += 20
    elif metrics.max_drawdown_pct < 25:
        score += 10

    # Convert to letter grade
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)