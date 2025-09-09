#!/usr/bin/env python3
"""
Quick Benchmark Runner

Simple script to benchmark trading system performance.

Usage:
    python run_benchmark.py                    # Test 100 symbols in 60 seconds
    python run_benchmark.py --quick            # Test 20 symbols in 15 seconds  
    python run_benchmark.py --symbols 50       # Test 50 symbols
    python run_benchmark.py --stress-test      # Test 200 symbols (stress test)
"""

import sys
import argparse
from src.utils.performance_benchmark import SystemBenchmarker
from src.utils.logging_config import logger


def run_benchmark_suite(symbols: int, target_time: float, save_file: str = None):
    """Run the benchmark suite with specified parameters."""
    
    print(f"\n🚀 Starting Performance Benchmark")
    print(f"📊 Testing {symbols} symbols")
    print(f"⏱️  Target time: {target_time} seconds")
    print(f"{'='*50}")
    
    try:
        benchmarker = SystemBenchmarker(target_time)
        results = benchmarker.run_full_benchmark(symbols)
        benchmarker.print_benchmark_report(results)
        
        if save_file:
            benchmarker.save_benchmark_results(results, save_file)
            print(f"\n📁 Results saved to: {save_file}")
        
        # Check if system meets requirements
        all_passed = all(result.success for result in results.values())
        
        if all_passed:
            print(f"\n✅ SUCCESS: System can handle {symbols} symbols within {target_time}s")
            return 0
        else:
            print(f"\n❌ FAILED: System cannot handle {symbols} symbols within {target_time}s")
            failed_components = [name for name, result in results.items() if not result.success]
            print(f"   Failed components: {', '.join(failed_components)}")
            return 1
            
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        print(f"❌ Benchmark failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Trading System Performance Benchmark")
    
    # Predefined test scenarios
    parser.add_argument('--quick', action='store_true',
                       help='Quick test: 20 symbols in 15 seconds')
    parser.add_argument('--standard', action='store_true', 
                       help='Standard test: 100 symbols in 60 seconds (default)')
    parser.add_argument('--stress-test', action='store_true',
                       help='Stress test: 200 symbols in 120 seconds')
    
    # Custom parameters
    parser.add_argument('--symbols', type=int,
                       help='Custom number of symbols to test')
    parser.add_argument('--target-time', type=float,
                       help='Custom target time in seconds')
    parser.add_argument('--save', type=str,
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Determine test parameters
    if args.quick:
        symbols, target_time = 20, 15.0
        test_name = "Quick Test"
    elif args.stress_test:
        symbols, target_time = 200, 120.0
        test_name = "Stress Test"
    elif args.symbols and args.target_time:
        symbols, target_time = args.symbols, args.target_time
        test_name = "Custom Test"
    elif args.symbols:
        symbols = args.symbols
        target_time = symbols * 0.6  # 0.6 seconds per symbol
        test_name = "Custom Symbol Count"
    else:
        # Default: Standard test
        symbols, target_time = 100, 60.0
        test_name = "Standard Test"
    
    print(f"🎯 Running {test_name}")
    
    return run_benchmark_suite(symbols, target_time, args.save)


if __name__ == "__main__":
    sys.exit(main())