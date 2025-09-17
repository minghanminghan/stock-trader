"""
Backtesting Module

Provides components for historical trading simulation and strategy validation.

Components:
- BacktestDataStream: Historical data replay with LiveDataStream interface
- SimulatedBroker: Portfolio simulation with AlpacaBroker interface
- Performance analytics and reporting tools
"""

from .data_stream import (
    BacktestDataStream,
    BacktestConfig,
    BacktestMode,
    create_backtest_stream
)

from .simulated_broker import (
    SimulatedBroker,
    Trade,
    Position,
    SimulatedAccount,
    OrderStatus
)

from .analytics import (
    BacktestAnalyzer,
    PerformanceMetrics,
    OptimalStrategy,
    run_backtest_analysis
)

__all__ = [
    # Data stream
    'BacktestDataStream',
    'BacktestConfig',
    'BacktestMode',
    'create_backtest_stream',

    # Broker simulation
    'SimulatedBroker',
    'Trade',
    'Position',
    'SimulatedAccount',
    'OrderStatus',

    # Analytics
    'BacktestAnalyzer',
    'PerformanceMetrics',
    'OptimalStrategy',
    'run_backtest_analysis'
]