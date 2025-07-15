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
        
        # Create analysis directory with date
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        analysis_dir = os.path.join('analysis', analysis_date)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save analysis summary text file
        summary_file = os.path.join(analysis_dir, 'analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== STOCK SCREENING OVERLAP ANALYSIS SUMMARY ===\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            preset_names = list(analysis_results['preset_results'].keys())
            f.write(f"Presets analyzed: {', '.join(preset_names)}\n")
            f.write(f"Total unique stocks across all presets: {analysis_results.get('total_unique_stocks', 0)}\n")
            
            # Count stocks appearing in 2+ presets
            overlap_groups = analysis_results.get('overlap_groups', {})
            overlap_count = sum(len(symbols) for count, symbols in overlap_groups.items() if count >= 2)
            f.write(f"Stocks appearing in 2+ presets: {overlap_count}\n\n")
            
            if overlap_count > 0:
                max_overlap = max(count for count in overlap_groups.keys() if count >= 2) if overlap_groups else 0
                f.write(f"Maximum preset overlap: {max_overlap}\n\n")
                
                f.write("=== TOP 10 CANDIDATES (by preset count and avg score) ===\n")
                # Create overlap stock list for summary
                all_overlap_stocks = []
                for count in sorted(overlap_groups.keys(), reverse=True):
                    if count >= 2:
                        for symbol in overlap_groups[count]:
                            # Get stock data from any preset it appears in
                            for preset_data in analysis_results['preset_results'].values():
                                if symbol in preset_data['symbols']:
                                    stock_results = preset_data['results']
                                    stock_info = next((s for s in stock_results if s['symbol'] == symbol), None)
                                    if stock_info:
                                        all_overlap_stocks.append({
                                            'symbol': symbol,
                                            'preset_count': count,
                                            'sector': stock_info.get('sector', 'N/A')
                                        })
                                        break
                
                for stock in all_overlap_stocks[:10]:
                    f.write(f"{stock['symbol']:<6} - {stock['preset_count']} presets, "
                           f"avg score: N/A, "
                           f"sector: {stock.get('sector', 'N/A')}\n")
        
        # Save overlap analysis TSV file
        overlap_file = os.path.join(analysis_dir, 'overlap_analysis.tsv')
        
        # Create overlap data in the format expected by export_overlap_analysis
        overlap_data = []
        overlap_groups = analysis_results.get('overlap_groups', {})
        preset_results = analysis_results.get('preset_results', {})
        
        for count in sorted(overlap_groups.keys(), reverse=True):
            if count >= 2:  # Only include stocks in 2+ presets
                for symbol in overlap_groups[count]:
                    stock_data = {'symbol': symbol, 'preset_count': count, 'presets': []}
                    
                    # Find which presets this symbol appears in and collect scores/ranks
                    for preset_name, preset_data in preset_results.items():
                        if symbol in preset_data['symbols']:
                            stock_data['presets'].append(preset_name)
                            # Get stock details from results
                            stock_results = preset_data['results']
                            stock_info = next((s for s in stock_results if s['symbol'] == symbol), None)
                            if stock_info:
                                # Add preset-specific score and rank
                                rank = preset_data['symbols'].index(symbol) + 1
                                stock_data[f'{preset_name}_score'] = stock_info.get('screening_score', 0)
                                stock_data[f'{preset_name}_rank'] = rank
                                
                                # Copy fundamental data (only need to do this once)
                                if 'sector' not in stock_data:
                                    for field in ['pe_ratio', 'pb_ratio', 'roe', 'market_cap', 'sector',
                                                'return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y', 'return_5y',
                                                'revenue_growth', 'earnings_growth', 'profit_margin', 'dividend_yield',
                                                'volatility', 'beta', 'current_ratio', 'debt_to_equity']:
                                        stock_data[field] = stock_info.get(field, '')
                    
                    # Calculate average score and rank
                    scores = [v for k, v in stock_data.items() if k.endswith('_score') and isinstance(v, (int, float))]
                    ranks = [v for k, v in stock_data.items() if k.endswith('_rank') and isinstance(v, (int, float))]
                    
                    stock_data['avg_score'] = sum(scores) / len(scores) if scores else 0
                    stock_data['avg_rank'] = sum(ranks) / len(ranks) if ranks else 0
                    stock_data['best_rank'] = min(ranks) if ranks else 0
                    
                    overlap_data.append(stock_data)
        
        if overlap_data:
            screener.export_overlap_analysis(overlap_data, overlap_file)
        
        print(f"\n💾 Analysis results saved to {analysis_dir}/")
        print(f"   - Summary: analysis_summary.txt")
        print(f"   - Overlap details: overlap_analysis.tsv")
        
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