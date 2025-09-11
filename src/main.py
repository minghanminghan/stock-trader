#!/usr/bin/env python3
"""
Trading System Orchestrator

Lightweight coordinator that brings together all trading components:
- Real-time data streaming (LiveDataStream)
- ML prediction pipeline (SignalGenerator)
- Order execution (AlpacaBroker)

Features:
- Clear logging and monitoring
- Graceful degradation on component failures
- Health checks and automatic recovery
- Clean shutdown handling
"""

import time
import signal
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from src.alpaca.data_stream import LiveDataStream, DataStreamMode
from src.alpaca.broker import AlpacaBroker
from src.trading.signal_generation import SignalGenerator, create_signal_generator
from src.trading.strategy import StrategyConfig
from src.models._utils.feature_engineering import compute_features
from src.config import TICKERS, LSTM_CONFIG
from src.utils.logging_config import logger


@dataclass
class OrchestratorConfig:
    """Configuration for the trading orchestrator."""
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: TICKERS)
    paper_trading: bool = True
    
    # Timing configuration
    signal_generation_interval: int = 60  # seconds
    data_buffer_minutes: int = 120  # minutes of data to maintain
    min_data_points: int = 60  # minimum data points before generating signals
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    max_consecutive_failures: int = 5
    data_staleness_threshold: int = 300  # seconds
    
    # Model configuration
    lstm_model_path: str = "src/models/lstm/weights/best_lstm_model.pth"
    strategy_type: str = "momentum"
    
    # Strategy parameters
    strategy_config: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        symbols=TICKERS,
        max_position_size=1000.0,
        buy_threshold=0.6,
        sell_threshold=0.6,
        max_open_positions=3,
        max_daily_trades=20
    ))


@dataclass
class SystemHealth:
    """Tracks overall system health."""
    data_stream_healthy: bool = False
    broker_healthy: bool = False
    signal_generator_healthy: bool = False
    last_signal_generation: Optional[datetime] = None
    last_data_received: Optional[datetime] = None
    consecutive_failures: int = 0
    total_signals_generated: int = 0
    total_orders_placed: int = 0


class TradingOrchestrator:
    """
    Main trading system orchestrator.
    
    Coordinates data streaming, signal generation, and order execution
    with comprehensive health monitoring and graceful degradation.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize trading orchestrator.
        
        Args:
            config: OrchestratorConfig with system parameters
        """
        self.config = config
        self.running = False
        self.health = SystemHealth()
        
        # # Component initialization
        # self.data_stream: LiveDataStream = None
        # self.broker: AlpacaBroker = None
        # self.signal_generator: SignalGenerator = None
        
        # # Threading
        # self.signal_thread: threading.Thread = None
        # self.health_thread: threading.Thread = None
        self.shutdown_event = threading.Event()
        
        # Data management
        self.latest_market_data: Dict[str, pd.DataFrame] = {}
        self.data_lock = threading.Lock()
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.stats: dict[str, int | float] = {
            'uptime_seconds': 0,
            'data_points_processed': 0,
            'signals_generated': 0,
            'orders_executed': 0,
            'component_restarts': 0,
            'degraded_operations': 0
        }
        
        logger.info("TradingOrchestrator initialized")
    
    def initialize_components(self) -> bool:
        """
        Initialize all trading system components.
        
        Returns:
            True if all components initialized successfully
        """
        logger.info("Initializing trading system components...")
        
        try:
            # Initialize data stream
            self.data_stream = LiveDataStream(
                buffer_size=self.config.data_buffer_minutes,
                max_retries=3,
                fallback_interval=30
            )
            self.data_stream.add_subscriber(self._on_new_data)
            logger.info("✓ Data stream initialized")
            
            # Initialize broker
            self.broker = AlpacaBroker(paper=self.config.paper_trading)
            account = self.broker.get_account()
            if account:
                logger.info(f"✓ Broker initialized - Portfolio: ${account['portfolio_value']:.2f}")
                self.health.broker_healthy = True
            else:
                logger.warning("Broker connected but account info unavailable")
            
            # Initialize signal generator
            self.signal_generator = create_signal_generator(
                broker=self.broker,
                strategy_config=self.config.strategy_config,
                strategy_type=self.config.strategy_type,
                model_path=self.config.lstm_model_path
            )
            logger.info("✓ Signal generator initialized")
            self.health.signal_generator_healthy = True
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def start(self) -> None:
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system already running")
            return
        
        logger.info("=" * 60)
        logger.info("STARTING TRADING SYSTEM")
        logger.info("=" * 60)
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components, aborting startup")
            return
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        try:
            # Start data stream
            logger.info(f"Starting data stream for symbols: {self.config.symbols}")
            self.data_stream.start_stream(self.config.symbols)
            self.health.data_stream_healthy = True
            
            # Start background threads
            self._start_signal_generation_thread()
            self._start_health_monitoring_thread()
            
            logger.info("Trading system started successfully")
            logger.info(f"Paper trading: {self.config.paper_trading}")
            logger.info(f"Signal generation interval: {self.config.signal_generation_interval}s")
            logger.info(f"Monitoring {len(self.config.symbols)} symbols")
            
            # Main loop - just monitor and log status
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading system gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping trading system...")
        self.running = False
        self.shutdown_event.set()
        
        # Stop components
        if self.data_stream:
            self.data_stream.stop_stream()
        
        # Wait for threads to finish
        threads = [self.signal_thread, self.health_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        # Log final statistics
        self._log_final_statistics()
        
        logger.info("Trading system stopped")
    
    def _start_signal_generation_thread(self) -> None:
        """Start signal generation thread."""
        self.signal_thread = threading.Thread(
            target=self._signal_generation_loop,
            name="SignalGeneration",
            daemon=True
        )
        self.signal_thread.start()
        logger.info("Signal generation thread started")
    
    def _start_health_monitoring_thread(self) -> None:
        """Start health monitoring thread."""
        self.health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            name="HealthMonitoring",
            daemon=True
        )
        self.health_thread.start()
        logger.info("Health monitoring thread started")
    
    def _signal_generation_loop(self) -> None:
        """Main signal generation loop."""
        logger.info("Signal generation loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check if we have sufficient data
                if self._has_sufficient_data():
                    market_data = self._prepare_market_data()
                    
                    if market_data is not None and not market_data.empty:
                        logger.info(f"Generating signals with {len(market_data)} data points")
                        
                        # Generate and execute signals
                        results = self.signal_generator.generate_and_execute_signals(market_data)
                        
                        # Update statistics
                        self.health.last_signal_generation = datetime.now(timezone.utc)
                        self.health.total_signals_generated += results.get('signals_generated', 0)
                        self.health.total_orders_placed += results.get('orders_placed', 0)
                        
                        self.stats['signals_generated'] += results.get('signals_generated', 0)
                        self.stats['orders_executed'] += results.get('orders_successful', 0)
                        
                        # Log results
                        if results.get('signals_generated', 0) > 0:
                            logger.info(f"Signal generation complete: "
                                      f"{results['signals_generated']} signals, "
                                      f"{results['orders_placed']} orders placed, "
                                      f"{results['orders_successful']} successful")
                        else:
                            logger.debug("No signals generated this cycle")
                        
                        # Log any errors
                        if results.get('errors'):
                            for error in results['errors']:
                                logger.warning(f"Signal generation error: {error}")
                        
                        self.health.consecutive_failures = 0
                    else:
                        logger.debug("Insufficient market data for signal generation")
                else:
                    logger.debug("Waiting for sufficient data...")
                
                # Wait for next cycle
                self.shutdown_event.wait(self.config.signal_generation_interval)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                self.health.consecutive_failures += 1
                
                if self.health.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.error("Max consecutive failures reached, attempting component restart")
                    self._attempt_component_recovery()
                
                # Back off on errors
                self.shutdown_event.wait(min(30, self.config.signal_generation_interval))
    
    def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                self._check_system_health()
                self._update_statistics()
                self._log_status()
                
                self.shutdown_event.wait(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                self.shutdown_event.wait(30)
    
    def _main_loop(self) -> None:
        """Main orchestrator loop."""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def _on_new_data(self, symbol: str, bar_data: Dict) -> None:
        """Handle new data from data stream."""
        try:
            with self.data_lock:
                self.health.last_data_received = datetime.now(timezone.utc)
                self.stats['data_points_processed'] += 1
                
                # Convert to DataFrame format and store
                if symbol not in self.latest_market_data:
                    self.latest_market_data[symbol] = pd.DataFrame()
                
                # Add new bar to symbol data
                new_row = pd.DataFrame([{
                    'symbol': symbol,
                    'timestamp': pd.to_datetime(bar_data['timestamp']),
                    'open': bar_data['open'],
                    'high': bar_data['high'],
                    'low': bar_data['low'],
                    'close': bar_data['close'],
                    'volume': bar_data['volume']
                }])
                
                self.latest_market_data[symbol] = pd.concat([
                    self.latest_market_data[symbol], new_row
                ], ignore_index=True)
                
                # Keep only recent data
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.config.data_buffer_minutes)
                self.latest_market_data[symbol] = self.latest_market_data[symbol][
                    self.latest_market_data[symbol]['timestamp'] >= cutoff_time
                ]
                
        except Exception as e:
            logger.error(f"Error processing new data for {symbol}: {e}")
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for signal generation."""
        with self.data_lock:
            if not self.latest_market_data:
                return False
            
            for symbol in self.config.symbols:
                if symbol not in self.latest_market_data:
                    return False
                if len(self.latest_market_data[symbol]) < self.config.min_data_points:
                    return False
            
            return True
    
    def _prepare_market_data(self) -> Optional[pd.DataFrame]:
        """Prepare market data for signal generation."""
        try:
            with self.data_lock:
                if not self.latest_market_data:
                    return None
                
                # Combine all symbol data
                all_data = []
                for symbol, df in self.latest_market_data.items():
                    if not df.empty:
                        df_copy = df.copy()
                        df_copy['symbol'] = symbol
                        all_data.append(df_copy)
                
                if not all_data:
                    return None
                
                # Combine and create MultiIndex
                market_data = pd.concat(all_data, ignore_index=True)
                market_data.set_index(['symbol', 'timestamp'], inplace=True)
                
                # Compute features if needed
                try:
                    market_data = compute_features(market_data)
                except Exception as e:
                    logger.warning(f"Feature computation failed, using raw data: {e}")
                
                return market_data
                
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return None
    
    def _check_system_health(self) -> None:
        """Check health of all system components."""
        current_time = datetime.now(timezone.utc)
        
        # Check data stream health
        if self.data_stream:
            stream_status = self.data_stream.get_stream_status()
            self.health.data_stream_healthy = (
                stream_status['is_running'] and 
                stream_status['mode'] != 'offline'
            )
        
        # Check data freshness
        if self.health.last_data_received:
            data_age = (current_time - self.health.last_data_received).total_seconds()
            if data_age > self.config.data_staleness_threshold:
                logger.warning(f"Data is stale: {data_age:.0f} seconds old")
        
        # Check broker health
        if self.broker:
            try:
                account = self.broker.get_account(use_cache=False)
                self.health.broker_healthy = account is not None
            except:
                self.health.broker_healthy = False
        
        # Check signal generator health
        signal_age = None
        if self.health.last_signal_generation:
            signal_age = (current_time - self.health.last_signal_generation).total_seconds()
        
        expected_interval = self.config.signal_generation_interval * 2  # Allow some slack
        if signal_age and signal_age > expected_interval:
            logger.warning(f"Signal generation delayed: {signal_age:.0f} seconds since last run")
    
    def _attempt_component_recovery(self) -> None:
        """Attempt to recover failed components."""
        logger.info("Attempting component recovery...")
        self.stats['component_restarts'] += 1
        
        try:
            # Try to reinitialize signal generator
            if not self.health.signal_generator_healthy:
                self.signal_generator = create_signal_generator(
                    broker=self.broker,
                    strategy_config=self.config.strategy_config,
                    strategy_type=self.config.strategy_type,
                    model_path=self.config.lstm_model_path
                )
                self.health.signal_generator_healthy = True
                logger.info("Signal generator recovered")
            
            # Reset failure counter
            self.health.consecutive_failures = 0
            
        except Exception as e:
            logger.error(f"Component recovery failed: {e}")
            self.stats['degraded_operations'] += 1
    
    def _update_statistics(self) -> None:
        """Update system statistics."""
        if self.start_time:
            self.stats['uptime_seconds'] = (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    def _log_status(self) -> None:
        """Log current system status."""
        uptime_hours = self.stats['uptime_seconds'] / 3600
        
        if uptime_hours > 0 and int(uptime_hours) % 1 == 0:  # Log every hour
            logger.info(
                f"System Status: "
                f"Uptime: {uptime_hours:.1f}h, "
                f"Data: {'✓' if self.health.data_stream_healthy else '✗'}, "
                f"Broker: {'✓' if self.health.broker_healthy else '✗'}, "
                f"Signals: {'✓' if self.health.signal_generator_healthy else '✗'}, "
                f"Generated: {self.stats['signals_generated']}, "
                f"Executed: {self.stats['orders_executed']}"
            )
    
    def _log_final_statistics(self) -> None:
        """Log final system statistics."""
        logger.info("=" * 60)
        logger.info("TRADING SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total uptime: {self.stats['uptime_seconds']/3600:.2f} hours")
        logger.info(f"Data points processed: {self.stats['data_points_processed']:,}")
        logger.info(f"Signals generated: {self.stats['signals_generated']}")
        logger.info(f"Orders executed: {self.stats['orders_executed']}")
        logger.info(f"Component restarts: {self.stats['component_restarts']}")
        logger.info(f"Degraded operations: {self.stats['degraded_operations']}")
        
        if self.signal_generator:
            signal_stats = self.signal_generator.get_signal_statistics()
            logger.info(f"Signal success rate: "
                       f"{signal_stats['execution_stats'].get('successful_orders', 0)}/"
                       f"{signal_stats['execution_stats'].get('total_orders', 1)}")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point for the trading system."""
    # Create configuration
    config = OrchestratorConfig(
        symbols=["SPY", "AAPL", "NVDA"],  # Start with just a few symbols
        paper_trading=True,
        signal_generation_interval=60,  # Generate signals every minute
        min_data_points=60,  # Need 1 hour of data minimum
    )
    
    # Create and start orchestrator
    orchestrator = TradingOrchestrator(config)
    
    try:
        orchestrator.start()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()