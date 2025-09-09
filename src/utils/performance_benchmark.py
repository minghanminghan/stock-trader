#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Stock Trading System

Tests whether the system can process 100 symbols within 1 minute target.
Measures CPU, memory, and processing time for each component.

Usage:
    python src/utils/performance_benchmark.py --symbols 100 --target-time 60
    python src/utils/performance_benchmark.py --quick-test
"""

import time
import psutil
import threading
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os

from src.live.data_stream import LiveDataStream
from src.live.signal_generator import ParallelSignalGenerator
from src.utils.parallel_features import compute_features_parallel
from src.trading.broker import AsyncAlpacaBroker
from src.utils.logging_config import logger


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    component: str
    symbols_count: int
    processing_time: float
    target_time: float
    success: bool
    throughput_symbols_per_sec: float
    cpu_usage_avg: float
    memory_usage_mb: float
    errors: List[str]
    metadata: Dict


class ResourceMonitor:
    """Monitors CPU and memory usage during benchmarks."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        
        def monitor():
            while self.monitoring:
                try:
                    cpu_percent = self.process.cpu_percent()
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    
                    time.sleep(self.interval)
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return averages."""
        self.monitoring = False
        
        if self.cpu_samples and self.memory_samples:
            return {
                'cpu_avg': np.mean(self.cpu_samples),
                'cpu_max': np.max(self.cpu_samples),
                'memory_avg': np.mean(self.memory_samples),
                'memory_max': np.max(self.memory_samples)
            }
        return {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}


class ComponentBenchmarker:
    """Benchmarks individual system components."""
    
    def __init__(self, target_time: float = 60.0):
        self.target_time = target_time
        self.resource_monitor = ResourceMonitor()
        
    def benchmark_data_ingestion(self, symbols: List[str]) -> BenchmarkResult:
        """Benchmark data ingestion for multiple symbols."""
        logger.info(f"Benchmarking data ingestion for {len(symbols)} symbols")
        
        errors = []
        start_time = time.time()
        
        self.resource_monitor.start_monitoring()
        
        try:
            # Initialize data stream (paper trading mode for testing)
            data_stream = LiveDataStream(buffer_size=100)
            
            # Simulate data collection for a short period
            data_stream.start_stream(symbols)
            time.sleep(2)  # Collect data for 2 seconds
            data_stream.stop_stream()

        except Exception as e:
            errors.append(f"Data ingestion failed: {e}")
            logger.error(f"Data ingestion error: {e}")
        
        processing_time = time.time() - start_time
        resource_stats = self.resource_monitor.stop_monitoring()
        
        return BenchmarkResult(
            component="data_ingestion",
            symbols_count=len(symbols),
            processing_time=processing_time,
            target_time=self.target_time,
            success=processing_time <= self.target_time and not errors,
            throughput_symbols_per_sec=len(symbols) / processing_time,
            cpu_usage_avg=resource_stats['cpu_avg'],
            memory_usage_mb=resource_stats['memory_avg'],
            errors=errors,
            metadata={'resource_stats': resource_stats}
        )
    
    def benchmark_feature_computation(self, symbols: List[str]) -> BenchmarkResult:
        """Benchmark parallel feature computation."""
        logger.info(f"Benchmarking feature computation for {len(symbols)} symbols")
        
        errors = []
        start_time = time.time()
        
        self.resource_monitor.start_monitoring()
        
        try:
            # Generate sample data for each symbol
            sample_data = {}
            for symbol in symbols:
                dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
                df = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 102,
                    'low': np.random.randn(1000).cumsum() + 98,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                }, index=dates)
                sample_data[symbol] = df
            
            # Create MultiIndex DataFrame from sample data
            all_dfs = []
            for symbol, df in sample_data.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy.set_index('symbol', append=True, inplace=True)
                df_copy = df_copy.reorder_levels([1, 0]).sort_index()  # symbol, timestamp
                all_dfs.append(df_copy)
            
            combined_df = pd.concat(all_dfs)
            
            # Compute features for all symbols using parallel function
            results = compute_features_parallel(combined_df, n_workers=4)
            
            if results is None or results.empty:
                errors.append("Feature computation returned no results")
            
        except Exception as e:
            errors.append(f"Feature computation failed: {e}")
            logger.error(f"Feature computation error: {e}")
        
        processing_time = time.time() - start_time
        resource_stats = self.resource_monitor.stop_monitoring()
        
        return BenchmarkResult(
            component="feature_computation",
            symbols_count=len(symbols),
            processing_time=processing_time,
            target_time=self.target_time,
            success=processing_time <= self.target_time and not errors,
            throughput_symbols_per_sec=len(symbols) / processing_time,
            cpu_usage_avg=resource_stats['cpu_avg'],
            memory_usage_mb=resource_stats['memory_avg'],
            errors=errors,
            metadata={'resource_stats': resource_stats, 'features_computed': len(results) if 'results' in locals() else 0}
        )
    
    def benchmark_signal_generation(self, symbols: List[str]) -> BenchmarkResult:
        """Benchmark ML signal generation."""
        logger.info(f"Benchmarking signal generation for {len(symbols)} symbols")
        
        errors = []
        start_time = time.time()
        
        self.resource_monitor.start_monitoring()
        
        try:
            # Initialize batch signal processor
            from src.live.signal_generator import BatchSignalProcessor
            signal_processor = BatchSignalProcessor()
            
            # Generate sample market data (format expected by BatchSignalProcessor)
            symbols_data = {}
            
            for symbol in symbols:
                # Create market data list of dicts
                market_data = []
                for i in range(100):  # 100 bars
                    timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
                    bar = {
                        'timestamp': timestamp,
                        'open': 100 + np.random.randn(),
                        'high': 102 + np.random.randn(),
                        'low': 98 + np.random.randn(),
                        'close': 100 + np.random.randn(),
                        'volume': np.random.randint(1000, 10000)
                    }
                    market_data.append(bar)
                symbols_data[symbol] = market_data
            
            # Generate signals for all symbols
            signals = signal_processor.generate_batch_signals(symbols_data)
            
            if not signals:
                errors.append("Signal generation returned no results")
            
        except Exception as e:
            errors.append(f"Signal generation failed: {e}")
            logger.error(f"Signal generation error: {e}")
        
        processing_time = time.time() - start_time
        resource_stats = self.resource_monitor.stop_monitoring()
        
        return BenchmarkResult(
            component="signal_generation",
            symbols_count=len(symbols),
            processing_time=processing_time,
            target_time=self.target_time,
            success=processing_time <= self.target_time and not errors,
            throughput_symbols_per_sec=len(symbols) / processing_time,
            cpu_usage_avg=resource_stats['cpu_avg'],
            memory_usage_mb=resource_stats['memory_avg'],
            errors=errors,
            metadata={'resource_stats': resource_stats, 'signals_generated': len(signals) if 'signals' in locals() else 0}
        )
    
    async def benchmark_order_processing(self, symbols: List[str]) -> BenchmarkResult:
        """Benchmark async order processing capabilities."""
        logger.info(f"Benchmarking order processing for {len(symbols)} symbols")
        
        errors = []
        start_time = time.time()
        
        self.resource_monitor.start_monitoring()
        
        try:
            # Initialize async broker (paper trading)
            async_broker = AsyncAlpacaBroker(paper_trading=True, max_workers=6)
            
            # Create sample orders for all symbols
            orders = []
            for symbol in symbols[:20]:  # Limit to 20 to avoid overwhelming paper trading
                orders.append({
                    'symbol': symbol,
                    'qty': 1,
                    'side': 'buy',
                    'order_type': 'market'
                })
            
            # Submit orders concurrently
            if orders:
                results = await async_broker.submit_multiple_orders_async(orders)
                successful_orders = sum(1 for r in results if r is not None)
                logger.info(f"Successfully processed {successful_orders}/{len(orders)} orders")
            
            async_broker.close()
            
        except Exception as e:
            errors.append(f"Order processing failed: {e}")
            logger.error(f"Order processing error: {e}")
        
        processing_time = time.time() - start_time
        resource_stats = self.resource_monitor.stop_monitoring()
        
        return BenchmarkResult(
            component="order_processing",
            symbols_count=len(symbols),
            processing_time=processing_time,
            target_time=self.target_time,
            success=processing_time <= self.target_time and not errors,
            throughput_symbols_per_sec=len(symbols) / processing_time,
            cpu_usage_avg=resource_stats['cpu_avg'],
            memory_usage_mb=resource_stats['memory_avg'],
            errors=errors,
            metadata={'resource_stats': resource_stats}
        )


class SystemBenchmarker:
    """Full system benchmark runner."""
    
    def __init__(self, target_time: float = 60.0):
        self.target_time = target_time
        self.component_benchmarker = ComponentBenchmarker(target_time)
    
    def get_sp100_symbols(self) -> List[str]:
        """Get S&P 100 symbols for testing."""
        # Sample of S&P 100 symbols for testing
        sp100_sample = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK.B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX', 'ADBE',
            'CRM', 'CMCSA', 'XOM', 'ABT', 'KO', 'PEP', 'COST', 'TMO', 'ACN', 'AVGO',
            'NKE', 'LLY', 'DHR', 'QCOM', 'TXN', 'NEE', 'WMT', 'BMY', 'UPS', 'PM',
            'LOW', 'IBM', 'HON', 'AMGN', 'INTC', 'SBUX', 'CVX', 'INTU', 'AMD', 'ORCL',
            'CAT', 'GE', 'AXP', 'MCD', 'GS', 'LMT', 'BLK', 'GILD', 'MMM', 'PFE',
            'BA', 'NOW', 'MO', 'SYK', 'TGT', 'ZTS', 'MDLZ', 'C', 'ISRG', 'CVS',
            'CI', 'DE', 'CHTR', 'FIS', 'AMT', 'ADP', 'VRTX', 'WFC', 'ANTM', 'USB',
            'BDX', 'TJX', 'CME', 'COP', 'SCHW', 'MU', 'AGN', 'MMC', 'D', 'NSC',
            'SO', 'ICE', 'DUK', 'PNC', 'AON', 'EQIX', 'CL', 'EMR', 'ITW', 'SHW'
        ]
        return sp100_sample
    
    def run_full_benchmark(self, symbol_count: int = 100) -> Dict[str, BenchmarkResult]:
        """Run complete system benchmark."""
        symbols = self.get_sp100_symbols()[:symbol_count]
        
        logger.info(f"Starting full system benchmark with {len(symbols)} symbols")
        logger.info(f"Target processing time: {self.target_time} seconds")
        
        results = {}
        
        # 1. Data Ingestion Benchmark
        logger.info("=== Benchmarking Data Ingestion ===")
        results['data_ingestion'] = self.component_benchmarker.benchmark_data_ingestion(symbols)
        
        # 2. Feature Computation Benchmark  
        logger.info("=== Benchmarking Feature Computation ===")
        results['feature_computation'] = self.component_benchmarker.benchmark_feature_computation(symbols)
        
        # 3. Signal Generation Benchmark
        logger.info("=== Benchmarking Signal Generation ===")
        results['signal_generation'] = self.component_benchmarker.benchmark_signal_generation(symbols)
        
        # 4. Order Processing Benchmark (async)
        logger.info("=== Benchmarking Order Processing ===")
        results['order_processing'] = asyncio.run(
            self.component_benchmarker.benchmark_order_processing(symbols)
        )
        
        return results
    
    def print_benchmark_report(self, results: Dict[str, BenchmarkResult]):
        """Print formatted benchmark results."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        print(f"Target Processing Time: {self.target_time}s")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_success = True
        total_time = 0
        
        for component, result in results.items():
            print(f"\n--- {result.component.upper().replace('_', ' ')} ---")
            print(f"Symbols Processed: {result.symbols_count}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Target Time: {result.target_time}s")
            print(f"Success: {'✓' if result.success else '✗'}")
            print(f"Throughput: {result.throughput_symbols_per_sec:.1f} symbols/sec")
            print(f"CPU Usage: {result.cpu_usage_avg:.1f}% (avg)")
            print(f"Memory Usage: {result.memory_usage_mb:.1f} MB (avg)")
            
            if result.errors:
                print(f"Errors: {', '.join(result.errors)}")
            
            total_time += result.processing_time
            overall_success &= result.success
        
        print(f"\n--- OVERALL RESULTS ---")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Overall Success: {'✓' if overall_success else '✗'}")
        print(f"System can handle {results[list(results.keys())[0]].symbols_count} symbols: {'YES' if overall_success else 'NO'}")
        print("="*80)
    
    def save_benchmark_results(self, results: Dict[str, BenchmarkResult], filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for component, result in results.items():
            json_results[component] = {
                'component': result.component,
                'symbols_count': result.symbols_count,
                'processing_time': result.processing_time,
                'target_time': result.target_time,
                'success': result.success,
                'throughput_symbols_per_sec': result.throughput_symbols_per_sec,
                'cpu_usage_avg': result.cpu_usage_avg,
                'memory_usage_mb': result.memory_usage_mb,
                'errors': result.errors,
                'metadata': result.metadata
            }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {filename}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    
    parser.add_argument('--symbols', type=int, default=100,
                       help='Number of symbols to test (default: 100)')
    parser.add_argument('--target-time', type=float, default=60.0,
                       help='Target processing time in seconds (default: 60)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 20 symbols')
    parser.add_argument('--save-results', type=str,
                       help='Save results to specified file')
    
    args = parser.parse_args()
    
    if args.quick_test:
        symbols_count = 20
        target_time = 15.0
    else:
        symbols_count = args.symbols
        target_time = args.target_time
    
    try:
        benchmarker = SystemBenchmarker(target_time)
        results = benchmarker.run_full_benchmark(symbols_count)
        benchmarker.print_benchmark_report(results)
        
        if args.save_results:
            benchmarker.save_benchmark_results(results, args.save_results)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())