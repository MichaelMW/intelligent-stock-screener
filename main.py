#!/usr/bin/env python3
"""
Standalone Stock Picker - Main Application
A minimal, functional stock screener with real data and performance tracking.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

# Add utils and screener to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'screener'))

from data_fetcher import DataFetcher
from stock_screener import StockScreener
from performance_tracker import PerformanceTracker

def main():
    parser = argparse.ArgumentParser(description='Standalone Stock Picker')
    parser.add_argument('--action', choices=['fetch', 'screen', 'backtest', 'full'], 
                       default='full', help='Action to perform')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to analyze')
    parser.add_argument('--preset', default='value_stocks', 
                       choices=['value_stocks', 'growth_stocks', 'quality_stocks', 'momentum_stocks',
                               'long_term_momentum', 'consistent_performers', 'low_volatility_growth', 'quality_dividend'],
                       help='Screening preset to use')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top stocks to return')
    parser.add_argument('--period', default='1y', help='Data period for fetching (1y, 2y, 5y)')
    parser.add_argument('--sp500', action='store_true', help='Use S&P 500 universe')
    parser.add_argument('--data-file', default='stock_data.json', help='Data file to use')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize components
    fetcher = DataFetcher(args.output_dir)
    screener = StockScreener(args.output_dir)
    tracker = PerformanceTracker(args.output_dir)
    
    print("=== STANDALONE STOCK PICKER ===")
    print(f"Action: {args.action}")
    print(f"Output directory: {args.output_dir}")
    
    if args.action in ['fetch', 'full']:
        print("\n=== FETCHING DATA ===")
        
        # Determine symbols to fetch
        if args.symbols:
            symbols = args.symbols
            print(f"Using provided symbols: {symbols}")
        elif args.sp500:
            print("Fetching S&P 500 symbols...")
            symbols = fetcher.get_sp500_symbols()
            print(f"Found {len(symbols)} S&P 500 symbols")
        else:
            # Default to a subset for testing
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'WMT']
            print(f"Using default test symbols: {symbols}")
        
        # Fetch data
        print(f"Fetching data for {len(symbols)} symbols...")
        data = fetcher.fetch_batch_data(symbols, args.period)
        
        if data:
            # Save data
            fetcher.save_data(data, args.data_file)
            fetcher.export_fundamentals_csv(data, 'fundamentals.csv')
            print(f"Data saved to {args.data_file}")
        else:
            print("No data fetched")
            return
    
    if args.action in ['screen', 'full']:
        print("\n=== SCREENING STOCKS ===")
        
        # Load data
        if not screener.load_data(args.data_file):
            print(f"Failed to load data from {args.data_file}")
            return
        
        # Get screening configuration
        presets = screener.get_preset_screens()
        if args.preset not in presets:
            print(f"Unknown preset: {args.preset}")
            print(f"Available presets: {list(presets.keys())}")
            return
        
        config = presets[args.preset]
        config['top_n'] = args.top_n
        
        print(f"Using preset: {args.preset}")
        print(f"Description: {config['description']}")
        
        # Run screening
        results = screener.screen_stocks(config)
        
        if results:
            print(f"\n=== SCREENING RESULTS ===")
            screener.print_results(args.top_n)
            
            # Export results
            output_file = f"{args.preset}_results_{datetime.now().strftime('%Y%m%d')}"
            screener.export_results(f"{output_file}.tsv", "tsv")
            
            # Store results for potential backtesting
            top_symbols = [stock['symbol'] for stock in results]
            
        else:
            print("No stocks found matching the criteria")
            return
    
    if args.action in ['backtest', 'full']:
        print("\n=== BACKTESTING PERFORMANCE ===")
        
        if args.action == 'backtest':
            # Load screening results if not from full run
            if not screener.load_data(args.data_file):
                print(f"Failed to load data from {args.data_file}")
                return
            
            # Run screening to get symbols
            presets = screener.get_preset_screens()
            config = presets[args.preset]
            config['top_n'] = args.top_n
            results = screener.screen_stocks(config)
            
            if not results:
                print("No stocks found for backtesting")
                return
            
            top_symbols = [stock['symbol'] for stock in results]
        
        # Calculate performance for last 6 months
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"Backtesting portfolio of {len(top_symbols)} stocks from {start_date}")
        print(f"Symbols: {top_symbols}")
        
        # Create portfolio
        portfolio_name = f"{args.preset}_portfolio"
        portfolio = tracker.create_portfolio(top_symbols, portfolio_name, start_date)
        
        # Calculate performance
        performance = tracker.calculate_portfolio_performance(portfolio_name)
        
        if performance:
            # Get benchmark performance
            benchmark = tracker.get_benchmark_performance('^GSPC', start_date)
            
            # Print performance report
            tracker.print_performance_report(performance, benchmark)
            
            # Save performance data
            perf_file = f"{args.preset}_performance_{datetime.now().strftime('%Y%m%d')}.json"
            tracker.save_performance_data(performance, perf_file)
            
            if benchmark:
                bench_file = f"sp500_benchmark_{datetime.now().strftime('%Y%m%d')}.json"
                tracker.save_performance_data(benchmark, bench_file)
        else:
            print("Failed to calculate portfolio performance")
    
    print("\n=== COMPLETE ===")
    print(f"All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()