#!/usr/bin/env python3
"""
Standalone Stock Picker - Main Application
A minimal, functional stock screener with real data and performance tracking.
"""
import argparse
import os
import sys
import json
from datetime import datetime, timedelta

# Add utils and screener to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'screener'))

from data_fetcher import DataFetcher
from stock_screener import StockScreener
from performance_tracker import PerformanceTracker

def load_available_presets():
    """Load available preset names from screening_presets.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'screening_presets.json')
    try:
        with open(config_path, 'r') as f:
            presets = json.load(f)
        return list(presets.keys())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load presets from {config_path}: {e}")
        print("Using fallback preset list")
        return ['deep_value', 'high_growth', 'fortress_balance_sheet']

def load_ticker_lists():
    """Load ticker lists from tickers.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'tickers.json')
    try:
        with open(config_path, 'r') as f:
            tickers = json.load(f)
        return tickers
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load tickers from {config_path}: {e}")
        return None

def main():
    # Load available presets dynamically
    available_presets = load_available_presets()
    default_preset = available_presets[0] if available_presets else 'deep_value'
    
    parser = argparse.ArgumentParser(description='Standalone Stock Picker')
    parser.add_argument('--action', choices=['fetch', 'screen', 'backtest', 'full', 'summarize', 'analyze'], 
                       default='full', help='Action to perform')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to analyze')
    parser.add_argument('--preset', default=default_preset, 
                       choices=available_presets,
                       help='Screening preset to use')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top stocks to return')
    parser.add_argument('--period', default='5y', help='Data period for fetching (1y, 2y, 5y) - 5y recommended for long-term analysis')
    parser.add_argument('--sp500', action='store_true', help='Use S&P 500 universe')
    parser.add_argument('--etf', action='store_true', help='Use ETF universe')
    parser.add_argument('--r2000', action='store_true', help='Use Russell 2000 universe')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--fetch-mode', choices=['batch', 'incremental'], default='batch',
                       help='Data fetch mode: batch (default, save all at once) or incremental (save per symbol)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads for parallel processing (default: auto)')
    
    args = parser.parse_args()
    
    # Initialize components
    fetcher = DataFetcher(args.output_dir)
    screener = StockScreener(args.output_dir)
    tracker = PerformanceTracker(args.output_dir)
    
    print("=== STANDALONE STOCK PICKER ===")
    print(f"Action: {args.action}")
    print(f"Output directory: {args.output_dir}")
    
    if args.action == 'summarize':
        print("\n=== RECALCULATING FUNDAMENTALS ===")
        print("Regenerating fundamentals.csv from existing price data (no new data fetching)...")
        
        # Check if database exists
        if not fetcher.storage.database_exists():
            print("❌ No database found. Run with --action fetch first to collect data.")
            return
        
        # Get database stats
        stats = fetcher.get_database_stats()
        print(f"📊 Database contains: {stats['symbols']} symbols, {stats['price_records']} price records")
        
        # Recalculate fundamentals from existing price data
        print("🔄 Recalculating return periods and metrics from existing price data...")
        fetcher.regenerate_fundamentals(max_workers=args.workers)
        
        # Export updated fundamentals CSV
        fetcher.export_fundamentals_csv(None, 'fundamentals.csv')
        
        print("✅ Fundamentals recalculation complete! (No new data was fetched)")
        return
    
    if args.action == 'analyze':
        print("\n=== ANALYZING ALL PRESETS ===")
        print("Running comprehensive analysis across all screening presets...")
        
        # Load data
        if not screener.load_data():
            print("Failed to load data from database or file")
            return
        
        # Run comprehensive analysis
        analysis_results = screener.analyze_all_presets(top_n=args.top_n)
        
        # Print results
        screener.print_analysis_results(analysis_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save preset results individually
        for preset_name, preset_data in analysis_results['preset_results'].items():
            if preset_data['count'] > 0:
                output_file = f"analysis_results_{preset_name}_{timestamp}.tsv"
                # Use the existing export method for each preset
                screener.screened_stocks = preset_data['results']
                screener.export_results(output_file, "tsv")
        
        # Save overlap analysis
        overlap_file = f"analysis_overlaps_{timestamp}.json"
        import json
        with open(os.path.join(args.output_dir, overlap_file), 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\n💾 Analysis results saved to {args.output_dir}")
        print(f"   - Individual preset results: analysis_results_[preset]_{timestamp}.tsv")
        print(f"   - Overlap analysis: {overlap_file}")
        
        return
    
    if args.action in ['fetch', 'full']:
        print("\n=== FETCHING DATA ===")
        
        symbols = []
        # Determine symbols to fetch
        if args.symbols:
            symbols += args.symbols
            print(f"Using provided symbols: {symbols}")
        if args.sp500:
            # Try to load from tickers.json first, fallback to fetcher method
            ticker_lists = load_ticker_lists()
            if ticker_lists and 'sp500' in ticker_lists:
                sp500_symbols = ticker_lists['sp500']
                symbols += sp500_symbols
                print(f"Using S&P 500 symbols from tickers.json: {len(sp500_symbols)} symbols")
            else:
                print("Fetching S&P 500 symbols from Wikipedia...")
                sp500_symbols = fetcher.get_sp500_symbols()
                symbols += sp500_symbols
                print(f"Found {len(sp500_symbols)} S&P 500 symbols")
        
        if args.etf:
            # Load ETF symbols from tickers.json
            ticker_lists = load_ticker_lists()
            if ticker_lists and 'etf' in ticker_lists:
                etf_symbols = ticker_lists['etf']
                symbols += etf_symbols
                print(f"Using ETF symbols from tickers.json: {len(etf_symbols)} symbols")
            else:
                print("Error: ETF ticker list not found in tickers.json")
                return
        
        if args.r2000:
            # Load Russell 2000 symbols from tickers.json
            ticker_lists = load_ticker_lists()
            if ticker_lists and 'r2000' in ticker_lists:
                r2000_symbols = ticker_lists['r2000']
                symbols += r2000_symbols
                print(f"Using Russell 2000 symbols from tickers.json: {len(r2000_symbols)} symbols")
            else:
                print("Error: Russell 2000 ticker list not found in tickers.json")
                return
        
        # If no symbols specified at all, use default test symbols
        if not symbols:
            symbols += ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'WMT']
            print(f"Using default test symbols: {symbols}")
        
        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))
        print(f"Total unique symbols to fetch: {len(symbols)}")
        
        # Fetch data based on selected mode
        use_incremental = args.fetch_mode == 'incremental'
        print(f"Fetching data for {len(symbols)} symbols using {args.fetch_mode} mode...")
        data = fetcher.fetch_batch_data(symbols, args.period, save_incrementally=use_incremental, max_workers=args.workers)
        
        if data:
            # Export fundamentals CSV for compatibility
            fetcher.export_fundamentals_csv(None, 'fundamentals.csv')  # Export from database
            
            # Show database statistics
            stats = fetcher.get_database_stats()
            print(f"Database stats: {stats['symbols']} symbols, {stats['price_records']} price records, {stats['db_size_mb']:.1f}MB")
        else:
            print("No data fetched")
            return
    
    if args.action in ['screen', 'full']:
        print("\n=== SCREENING STOCKS ===")
        
        # Load data (SQLite database by default, JSON fallback)
        if not screener.load_data():
            print("Failed to load data from database or file")
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
            if not screener.load_data():
                print("Failed to load data from database or file")
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