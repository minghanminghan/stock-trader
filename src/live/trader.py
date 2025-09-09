import asyncio
import time
from typing import List, Dict, Optional
from datetime import datetime
import signal
import sys
import traceback

from src.trading.broker import AsyncAlpacaBroker, BatchOrderManager
from src.trading.strategy import MomentumTradingStrategy, TradingDecision, Action
from src.trading.portfolio import Portfolio
from src.trading.risk_manager import RiskManager
from src.live.data_stream import LiveDataStream
from src.live.signal_generator import ParallelSignalGenerator, BatchSignalProcessor
from src.utils.parallel_features import StreamingFeatureComputer
from src.utils.logging_config import logger


class LiveTrader:
    """
    High-performance live trading system with parallelization optimizations.
    
    Optimizations:
    - Parallel signal generation across symbols
    - Async broker operations  
    - Batch order execution
    - Streaming feature computation
    - Concurrent risk checks
    """
    
    def __init__(self, 
                 symbols: List[str],
                 paper_trading: bool = True,
                 loop_interval: int = 60,
                 max_daily_trades: int = 100,
                 enable_batch_processing: bool = True):
        """
        Initialize optimized live trader.
        
        Args:
            symbols: List of symbols to trade
            paper_trading: Use paper trading environment
            loop_interval: Main loop interval in seconds
            max_daily_trades: Maximum trades per day
            enable_batch_processing: Use batch signal processing for better performance
        """
        self.symbols = symbols
        self.loop_interval = loop_interval
        self.max_daily_trades = max_daily_trades
        self.enable_batch_processing = enable_batch_processing
        
        # Initialize optimized components
        self.broker = AsyncAlpacaBroker(paper_trading=paper_trading, max_workers=8)
        self.batch_manager = BatchOrderManager(self.broker, rate_limit_delay=0.1)
        self.strategy = MomentumTradingStrategy()
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager()
        self.data_stream = LiveDataStream(buffer_size=300)  # Larger buffer for better features
        
        # Choose signal processing approach
        if enable_batch_processing and len(symbols) > 2:
            self.signal_processor = BatchSignalProcessor()
            logger.info("Using batch signal processing for optimal performance")
        else:
            self.signal_generator = ParallelSignalGenerator(max_workers=min(4, len(symbols)))
            logger.info("Using parallel signal generation")
        
        # Streaming feature computer for incremental feature updates
        self.feature_computer = StreamingFeatureComputer(max_window_size=300)
        
        # State tracking
        self.is_running = False
        self.should_stop = False
        self.daily_trade_count = 0
        self.last_heartbeat = datetime.now()
        self.errors_count = 0
        self.max_errors = 10
        
        # Performance metrics
        self.loop_times = []
        self.total_loops = 0
        self.signal_generation_times = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"LiveTrader initialized - Symbols: {symbols}, "
                   f"Batch processing: {enable_batch_processing}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.should_stop = True
    
    async def start_trading_async(self) -> None:
        """Start the optimized trading system."""
        if self.is_running:
            logger.warning("Trading system already running")
            return
        
        try:
            logger.info("=== Starting Live Trading System ===")
            
            # Pre-flight checks
            if not await self._preflight_checks_async():
                logger.error("Pre-flight checks failed, aborting")
                return
            
            # Initialize daily equity tracking
            account_info = await self.broker.get_account_balance_async()
            if account_info:
                self.risk_manager.set_daily_start_equity(account_info['equity'])
            
            # Start data stream
            self.data_stream.start_stream(self.symbols)
            
            # Wait for initial data
            await self._wait_for_initial_data_async()
            
            # Sync portfolio with broker
            await self._sync_portfolio_async()
            
            # Main trading loop
            self.is_running = True
            await self._run_optimized_trading_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in trading system: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self._shutdown_async()
    
    async def _preflight_checks_async(self) -> bool:
        """Run async pre-flight checks."""
        logger.info("Running pre-flight checks...")
        
        # Check market and account concurrently
        market_open_task = asyncio.create_task(self._check_market_open_async())
        account_task = self.broker.get_account_balance_async()
        
        is_market_open, account_info = await asyncio.gather(
            market_open_task, account_task, return_exceptions=True
        )
        
        if isinstance(account_info, Exception) or not account_info:
            logger.error("Cannot access account information")
            return False
        
        logger.info(f"Account equity: ${account_info['equity']:,.2f}")
        logger.info(f"Buying power: ${account_info['buying_power']:,.2f}")
        
        # Check signal processing is ready
        if self.enable_batch_processing:
            if not hasattr(self.signal_processor, 'predictor') or not self.signal_processor.predictor.model:
                logger.error("Batch signal processor not ready")
                return False
        else:
            model_status = self.signal_generator.get_model_status()
            if not model_status['model_loaded']:
                logger.error("ML model not loaded")
                return False
        
        logger.info("Pre-flight checks passed ✓")
        return True
    
    async def _check_market_open_async(self) -> bool:
        """Check if market is open asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.broker.is_market_open)
    
    async def _wait_for_initial_data_async(self, timeout_seconds: int = 60) -> None:
        """Wait for initial market data asynchronously."""
        logger.info("Waiting for initial market data...")
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            # Check data freshness for all symbols
            fresh_symbols = [
                symbol for symbol in self.symbols 
                if self.data_stream.is_data_fresh(symbol, max_age_minutes=10)
            ]
            
            if len(fresh_symbols) == len(self.symbols):
                logger.info("Initial market data ready for all symbols")
                return
            
            await asyncio.sleep(2)  # Check every 2 seconds
        
        logger.warning("Timeout waiting for market data, proceeding anyway")
    
    async def _sync_portfolio_async(self) -> None:
        """Sync portfolio with broker asynchronously."""
        logger.info("Syncing portfolio with broker...")
        
        broker_positions = await self.broker.get_all_positions_async()
        self.portfolio.update_all_positions_from_broker(broker_positions)
        
        summary = self.portfolio.get_portfolio_summary()
        logger.info(f"Portfolio synced: {summary['total_positions']} positions, "
                   f"${summary['portfolio_value']:,.2f} value")
    
    async def _run_optimized_trading_loop(self) -> None:
        """Main optimized trading loop with async operations."""
        logger.info("Starting optimized trading loop...")
        
        while not self.should_stop and self.errors_count < self.max_errors:
            loop_start = time.time()
            
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                self.total_loops += 1
                
                # Check trading conditions
                if not await self._should_continue_trading_async():
                    logger.debug("Trading conditions not met, skipping loop")
                    await self._sleep_until_next_loop_async(loop_start)
                    continue
                
                # Execute optimized trading cycle
                await self._execute_optimized_trading_cycle()
                
                # Reset error count on success
                self.errors_count = 0
                
            except Exception as e:
                self.errors_count += 1
                logger.error(f"Error in trading loop (count: {self.errors_count}): {e}")
                logger.error(traceback.format_exc())
                
                if self.errors_count >= self.max_errors:
                    logger.error("Max errors reached, stopping trading")
                    break
            
            # Performance tracking
            loop_time = time.time() - loop_start
            self.loop_times.append(loop_time)
            if len(self.loop_times) > 100:
                self.loop_times.pop(0)
            
            logger.debug(f"Loop {self.total_loops} completed in {loop_time:.3f}s")
            
            # Sleep until next iteration
            await self._sleep_until_next_loop_async(loop_start)
        
        logger.info("Optimized trading loop ended")
    
    async def _should_continue_trading_async(self) -> bool:
        """Check trading conditions asynchronously."""
        # Run checks concurrently
        market_check_task = self._check_market_open_async()
        account_task = self.broker.get_account_balance_async()
        
        is_market_open, account_info = await asyncio.gather(
            market_check_task, account_task, return_exceptions=True
        )
        
        if isinstance(is_market_open, Exception) or not is_market_open:
            return False
        
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        if not isinstance(account_info, Exception) and account_info:
            if not self.risk_manager.check_daily_loss_limit(account_info['equity']):
                return False
        
        return True
    
    async def _execute_optimized_trading_cycle(self) -> None:
        """Execute optimized trading cycle with parallel operations."""
        signal_start = time.time()
        
        # 1. Generate signals (optimized based on processing mode)
        if self.enable_batch_processing:
            signals_data = {}
            for symbol in self.symbols:
                market_data = self.data_stream.get_historical_bars(symbol, count=100)
                if market_data:
                    signals_data[symbol] = market_data
            
            if signals_data:
                signal_results = self.signal_processor.generate_batch_signals(signals_data)
                signals = list(signal_results.values())
            else:
                signals = []
        else:
            signals = self.signal_generator.generate_signals_for_symbols(
                self.symbols, self.data_stream
            )
        
        signal_time = time.time() - signal_start
        self.signal_generation_times.append(signal_time)
        
        # 2. Get strategy decisions
        decisions = self.strategy.rank_opportunities(signals, self.portfolio)
        
        # 3. Execute remaining operations concurrently
        account_task = self.broker.get_account_balance_async()
        prices_task = self._get_current_prices_async()
        positions_task = self.broker.get_positions_batch_async(self.symbols)
        
        account_info, current_prices, broker_positions = await asyncio.gather(
            account_task, prices_task, positions_task, return_exceptions=True
        )
        
        if isinstance(account_info, Exception) or not account_info:
            logger.error("Cannot get account info, skipping cycle")
            return
        
        # 4. Update portfolio with fresh data
        if not isinstance(current_prices, Exception) and current_prices:
            self.portfolio.update_market_prices(current_prices)
        
        # 5. Process risk checks and trading decisions concurrently
        risk_task = self._check_risk_exits_async(current_prices, account_info['equity'])
        decisions_task = self._process_trading_decisions_async(
            decisions, account_info['equity'], current_prices
        )
        
        await asyncio.gather(risk_task, decisions_task, return_exceptions=True)
        
        # 6. Execute queued orders in batch
        if self.batch_manager.pending_orders:
            await self.batch_manager.execute_batch(max_batch_size=8)
    
    async def _get_current_prices_async(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}
        for symbol in self.symbols:
            price = self.data_stream.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    async def _check_risk_exits_async(self, current_prices: Dict[str, float], 
                                     account_equity: float) -> None:
        """Check risk exits and queue orders asynchronously."""
        for symbol, price in current_prices.items():
            position = self.portfolio.get_position(symbol)
            if position is None:
                continue
            
            # Check stop loss
            if self.risk_manager.should_stop_out(symbol, price, self.portfolio):
                side = "sell" if position.qty > 0 else "buy"
                self.batch_manager.queue_order(
                    symbol, abs(position.qty), side, priority=0  # High priority for risk exits
                )
                logger.warning(f"Queued risk exit: {side.upper()} {abs(position.qty)} {symbol}")
            
            # Check take profit
            elif self.risk_manager.should_take_profit(symbol, price, self.portfolio):
                side = "sell" if position.qty > 0 else "buy"
                self.batch_manager.queue_order(
                    symbol, abs(position.qty), side, priority=1  # High priority
                )
                logger.info(f"Queued profit taking: {side.upper()} {abs(position.qty)} {symbol}")
    
    async def _process_trading_decisions_async(self, decisions: List[TradingDecision],
                                             account_equity: float, 
                                             current_prices: Dict[str, float]) -> None:
        """Process trading decisions and queue orders."""
        for decision in decisions:
            if decision.action == Action.HOLD:
                continue
            
            symbol = decision.symbol
            current_price = current_prices.get(symbol)
            
            if current_price is None:
                continue
            
            # Validate trade
            is_valid, reason = self.risk_manager.validate_trade(
                symbol, decision, account_equity, current_price, self.portfolio
            )
            
            if not is_valid:
                logger.debug(f"Trade rejected for {symbol}: {reason}")
                continue
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, decision, account_equity, current_price, self.portfolio
            )
            
            if position_size > 0:
                self.batch_manager.queue_order(
                    symbol, position_size, decision.action.value, priority=2
                )
                logger.debug(f"Queued order: {decision.action.value.upper()} {position_size} {symbol}")
    
    async def _sleep_until_next_loop_async(self, loop_start_time: float) -> None:
        """Sleep until next loop iteration."""
        elapsed = time.time() - loop_start_time
        sleep_time = max(0, self.loop_interval - elapsed)
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
    
    async def _shutdown_async(self) -> None:
        """Async graceful shutdown."""
        logger.info("Shutting down optimized trading system...")
        
        self.is_running = False
        
        # Execute any remaining orders
        if self.batch_manager.pending_orders:
            logger.info("Executing remaining orders before shutdown...")
            await self.batch_manager.execute_batch()
        
        # Stop data stream
        if hasattr(self, 'data_stream'):
            self.data_stream.stop_stream()
        
        # Final portfolio sync
        try:
            await self._sync_portfolio_async()
            summary = self.portfolio.get_portfolio_summary()
            logger.info(f"Final portfolio: {summary['total_positions']} positions, "
                       f"${summary['portfolio_value']:,.2f} value")
        except Exception as e:
            logger.error(f"Error during final sync: {e}")
        
        # Close async components
        self.broker.close()
        
        # Performance summary
        if self.loop_times:
            avg_loop_time = sum(self.loop_times) / len(self.loop_times)
            logger.info(f"Performance: {self.total_loops} loops, "
                       f"avg loop time: {avg_loop_time:.3f}s")
        
        if self.signal_generation_times:
            avg_signal_time = sum(self.signal_generation_times) / len(self.signal_generation_times)
            logger.info(f"Avg signal generation time: {avg_signal_time:.3f}s")
        
        logger.info("Optimized trading system shutdown complete")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        avg_loop_time = sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0
        avg_signal_time = (sum(self.signal_generation_times) / len(self.signal_generation_times) 
                          if self.signal_generation_times else 0)
        
        return {
            'total_loops': self.total_loops,
            'avg_loop_time': avg_loop_time,
            'avg_signal_time': avg_signal_time,
            'errors_count': self.errors_count,
            'daily_trade_count': self.daily_trade_count,
            'optimization_mode': 'batch' if self.enable_batch_processing else 'parallel'
        }


# Entry point for optimized trading
def run_live_trading(symbols: List[str], paper_trading: bool = True, 
                    enable_batch_processing: bool = True):
    """Run the live trading system."""
    trader = LiveTrader(
        symbols=symbols,
        paper_trading=paper_trading,
        enable_batch_processing=enable_batch_processing
    )
    
    try:
        asyncio.run(trader.start_trading_async())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    # Test optimized trader
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    run_live_trading(symbols, paper_trading=True)