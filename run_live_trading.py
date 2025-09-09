#!/usr/bin/env python3
"""
Live Trading System Runner

Example usage:
    # Paper trading (safe)
    python run_live_trading.py --paper

    # Live trading (real money - be careful!)
    python run_live_trading.py --live

    # Custom symbols and settings
    python run_live_trading.py --paper --symbols AAPL MSFT GOOGL --interval 30
"""

import argparse
import sys
import time
from typing import List

from src.live.trader import LiveTrader
from src.utils.logging_config import logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Trading System")
    
    # Trading mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--paper', action='store_true', 
                           help='Run in paper trading mode (safe)')
    mode_group.add_argument('--live', action='store_true',
                           help='Run in live trading mode (real money!)')
    
    # Symbols to trade
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'GOOGL', 'AMZN', 'NVDA'],
                       help='Stock symbols to trade (default: AAPL GOOGL AMZN NVDA)')
    
    # Trading parameters
    parser.add_argument('--interval', type=int, default=60,
                       help='Main loop interval in seconds (default: 60)')
    
    parser.add_argument('--max-trades', type=int, default=100,
                       help='Maximum trades per day (default: 100)')
    
    # Safety options
    parser.add_argument('--confirm-live', action='store_true',
                       help='Skip live trading confirmation (dangerous!)')
    
    return parser.parse_args()


def confirm_live_trading(symbols: List[str]) -> bool:
    """Confirm user wants to do live trading with real money."""
    print("\n" + "="*60)
    print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK ⚠️")
    print("="*60)
    print(f"You are about to start live trading with the following symbols:")
    print(f"  {', '.join(symbols)}")
    print("\nThis will place REAL orders with REAL money!")
    print("Losses can occur and may be substantial.")
    print("\nPlease confirm you understand the risks:")
    
    while True:
        response = input("Type 'I UNDERSTAND THE RISKS' to continue: ").strip()
        if response == "I UNDERSTAND THE RISKS":
            return True
        elif response.lower() in ['quit', 'exit', 'no', 'cancel']:
            return False
        else:
            print("Please type exactly 'I UNDERSTAND THE RISKS' or 'quit' to exit")


def print_status(trader: LiveTrader, interval: int = 300):
    """Print periodic status updates."""
    last_status_time = time.time()
    
    while trader.is_running:
        current_time = time.time()
        
        if current_time - last_status_time >= interval:
            status = trader.get_status()
            
            print(f"\n{'='*50}")
            print(f"Status Update - {time.strftime('%H:%M:%S')}")
            print(f"{'='*50}")
            print(f"Running: {status['is_running']}")
            print(f"Total Loops: {status['total_loops']}")
            print(f"Daily Trades: {status['daily_trade_count']}")
            print(f"Avg Loop Time: {status['avg_loop_time']:.2f}s")
            
            portfolio = status['portfolio_summary']
            print(f"Portfolio Value: ${portfolio['portfolio_value']:,.2f}")
            print(f"Unrealized P&L: ${portfolio['unrealized_pl']:,.2f}")
            print(f"Realized P&L: ${portfolio['realized_pl_today']:,.2f}")
            print(f"Open Positions: {portfolio['total_positions']}")
            
            risk_metrics = status['risk_metrics']
            if risk_metrics['daily_pl_pct'] != 0:
                print(f"Daily P&L: {risk_metrics['daily_pl_pct']:.2%}")
            
            last_status_time = current_time
        
        time.sleep(30)  # Check every 30 seconds


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Determine trading mode
    paper_trading = args.paper
    
    # Safety check for live trading
    if not paper_trading and not args.confirm_live:
        if not confirm_live_trading(args.symbols):
            print("Live trading cancelled by user")
            sys.exit(0)
    
    # Log trading mode
    mode_str = "PAPER TRADING" if paper_trading else "LIVE TRADING"
    logger.info(f"Starting {mode_str} mode")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Loop interval: {args.interval} seconds")
    
    try:
        # Initialize trader
        trader = LiveTrader(
            symbols=args.symbols,
            paper_trading=paper_trading,
            loop_interval=args.interval,
            max_daily_trades=args.max_trades
        )
        
        # Start status monitoring in background
        import threading
        status_thread = threading.Thread(target=print_status, args=(trader,))
        status_thread.daemon = True
        status_thread.start()
        
        # Start trading (this blocks) - check if async method exists
        if hasattr(trader, 'start_trading_async'):
            import asyncio
            asyncio.run(trader.start_trading_async())
        else:
            trader.start_trading()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()