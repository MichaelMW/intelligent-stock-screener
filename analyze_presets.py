#!/usr/bin/env python3
"""
Advanced Stock Screening Analysis Tool
Analyzes overlap between different screening presets and identifies high-quality candidates.
Auto-detects missing preset results and runs screening as needed.
"""
import pandas as pd
import json
import glob
import argparse
from collections import defaultdict
from datetime import datetime
import os
import sys
import subprocess

def load_all_preset_names():
    """Load all preset names from the configuration file"""
    try:
        with open('screener/screening_presets.json', 'r') as f:
            presets_config = json.load(f)
        return list(presets_config.keys())
    except Exception as e:
        print(f"Error loading preset configuration: {e}")
        # Fallback to known presets
        return ['value_stocks', 'growth_stocks', 'quality_stocks', 'momentum_stocks', 
                'long_term_momentum', 'consistent_performers', 'low_volatility_growth', 'quality_dividend']

def load_screening_results():
    """Load all screening results from TSV files"""
    results = {}
    tsv_files = glob.glob('analysis/temp_results/*_results_*.tsv')
    
    for file in tsv_files:
        # Extract preset name from filename
        filename = os.path.basename(file)
        preset_name = filename.split('_results_')[0]
        
        try:
            df = pd.read_csv(file, sep='\t')
            if not df.empty:
                results[preset_name] = df
                print(f"Loaded {len(df)} stocks from {preset_name}")
            else:
                print(f"Warning: Empty results for {preset_name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def get_today_date_string():
    """Get today's date in YYYYMMDD format"""
    return datetime.now().strftime('%Y%m%d')

def create_analysis_output_dir():
    """Create organized analysis output directory structure"""
    today = datetime.now().strftime('%Y-%m-%d')
    analysis_dir = f"analysis/{today}"
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir

def check_missing_presets(all_presets, date_str=None, force_regenerate=True):
    """Check which preset result files are missing for the given date"""
    if date_str is None:
        date_str = get_today_date_string()
    
    if force_regenerate:
        # Always regenerate temp files to reflect latest preset configurations
        print("🔄 Force regenerating all temp results to reflect latest preset configurations...")
        clear_temp_results()
        return all_presets, []  # All presets are "missing" (need regeneration)
    
    missing_presets = []
    existing_presets = []
    
    for preset in all_presets:
        result_file = f"analysis/temp_results/{preset}_results_{date_str}.tsv"
        if os.path.exists(result_file):
            existing_presets.append(preset)
        else:
            missing_presets.append(preset)
    
    return missing_presets, existing_presets

def clear_temp_results():
    """Clear all temp result files"""
    import glob
    temp_files = glob.glob('analysis/temp_results/*_results_*.tsv')
    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove {file}: {e}")

def run_missing_screenings(missing_presets, data_file="stock_data.json"):
    """Automatically run screening for missing presets"""
    if not missing_presets:
        print("All preset screenings are up to date!")
        return True
    
    print(f"\n=== AUTO-RUNNING MISSING PRESETS ===")
    print(f"Missing presets: {', '.join(missing_presets)}")
    
    # Check if data file exists
    if not os.path.exists(f"data/{data_file}"):
        print(f"Data file {data_file} not found. Please run data fetching first:")
        print(f"python main.py --action fetch --sp500")
        return False
    
    success_count = 0
    for preset in missing_presets:
        print(f"\nRunning screening for preset: {preset}")
        try:
            # Run the screening using main.py
            cmd = [sys.executable, "main.py", "--action", "screen", "--preset", preset, "--data-file", data_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Successfully completed screening for {preset}")
                success_count += 1
            else:
                print(f"✗ Failed to run screening for {preset}")
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout running screening for {preset}")
        except Exception as e:
            print(f"✗ Error running screening for {preset}: {e}")
    
    print(f"\nCompleted {success_count}/{len(missing_presets)} preset screenings")
    return success_count == len(missing_presets)

def find_overlapping_stocks(results, min_presets=2):
    """Find stocks that appear in multiple presets"""
    stock_counts = defaultdict(set)
    stock_details = defaultdict(dict)
    
    # Count appearances and collect details
    for preset_name, df in results.items():
        for _, row in df.iterrows():
            symbol = row['symbol']
            stock_counts[symbol].add(preset_name)
            stock_details[symbol][preset_name] = {
                'rank': row.get('Unnamed: 0', getattr(row, 'name', 0)) + 1,  # CSV index as rank
                'score': row.get('screening_score', 0),
                'pe_ratio': row.get('pe_ratio', None),
                'pb_ratio': row.get('pb_ratio', None),
                'roe': row.get('roe', None),
                'market_cap': row.get('market_cap', None),
                'sector': row.get('sector', None),
                'return_1m': row.get('return_1m', None),
                'return_3m': row.get('return_3m', None),
                'revenue_growth': row.get('revenue_growth', None),
                'earnings_growth': row.get('earnings_growth', None),
                'profit_margin': row.get('profit_margin', None),
                'dividend_yield': row.get('dividend_yield', None)
            }
    
    # Filter stocks appearing in at least min_presets
    overlapping_stocks = {
        symbol: presets 
        for symbol, presets in stock_counts.items() 
        if len(presets) >= min_presets
    }
    
    return overlapping_stocks, stock_details

def analyze_quality_preset_issue(results):
    """Analyze why quality_stocks preset returns no results"""
    print("\n=== QUALITY PRESET ANALYSIS ===")
    
    # Load the original data to see what's happening
    try:
        with open('data/stock_data.json', 'r') as f:
            stock_data = json.load(f)
        
        # Analyze the quality filters
        quality_filters = {
            'roe_gt_15': 0,
            'debt_to_equity_lt_50': 0,
            'current_ratio_gt_1': 0,
            'profit_margin_gt_10': 0,
            'market_cap_gt_5B': 0
        }
        
        total_stocks = len(stock_data)
        stocks_with_data = 0
        
        for symbol, data in stock_data.items():
            fundamentals = data.get('fundamentals', {})
            
            if fundamentals:
                stocks_with_data += 1
                
                # Check each filter
                roe = fundamentals.get('roe')
                if roe is not None and roe > 0.15:
                    quality_filters['roe_gt_15'] += 1
                
                debt_eq = fundamentals.get('debt_to_equity')
                if debt_eq is not None and debt_eq < 0.5:
                    quality_filters['debt_to_equity_lt_50'] += 1
                
                current_ratio = fundamentals.get('current_ratio')
                if current_ratio is not None and current_ratio > 1.0:
                    quality_filters['current_ratio_gt_1'] += 1
                
                profit_margin = fundamentals.get('profit_margin')
                if profit_margin is not None and profit_margin > 0.1:
                    quality_filters['profit_margin_gt_10'] += 1
                
                market_cap = fundamentals.get('market_cap')
                if market_cap is not None and market_cap > 5e9:
                    quality_filters['market_cap_gt_5B'] += 1
        
        print(f"Total stocks: {total_stocks}")
        print(f"Stocks with fundamental data: {stocks_with_data}")
        print("\nFilter Analysis:")
        for filter_name, count in quality_filters.items():
            percentage = (count / stocks_with_data) * 100 if stocks_with_data > 0 else 0
            print(f"  {filter_name}: {count} stocks ({percentage:.1f}%)")
        
        # Suggest alternative quality criteria
        print("\nSuggested Quality Filter Adjustments:")
        print("  - ROE > 10% instead of 15%")
        print("  - Debt/Equity < 1.0 instead of 0.5")
        print("  - Market Cap > $1B instead of $5B")
        print("  - Profit Margin > 5% instead of 10%")
        
    except Exception as e:
        print(f"Error analyzing quality preset: {e}")

def create_overlap_report(overlapping_stocks, stock_details, min_presets=2):
    """Create a detailed report of overlapping stocks"""
    print(f"\n=== STOCKS APPEARING IN {min_presets}+ PRESETS ===")
    
    # Sort by number of presets (descending), then by average score
    sorted_stocks = []
    for symbol, presets in overlapping_stocks.items():
        preset_count = len(presets)
        avg_score = sum(stock_details[symbol][preset]['score'] for preset in presets) / preset_count
        sorted_stocks.append((symbol, preset_count, avg_score, presets))
    
    sorted_stocks.sort(key=lambda x: (-x[1], -x[2]))  # Sort by preset count desc, then avg score desc
    
    if not sorted_stocks:
        print(f"No stocks found appearing in {min_presets}+ presets")
        return
    
    print(f"Found {len(sorted_stocks)} stocks appearing in {min_presets}+ presets:\n")
    
    # Header
    print(f"{'Symbol':<8} {'Presets':<3} {'Avg Score':<10} {'Presets List':<40} {'Sector':<20} {'Market Cap':<12}")
    print("-" * 100)
    
    # Stock details
    for symbol, preset_count, avg_score, presets in sorted_stocks:
        # Get basic info from first preset
        first_preset = list(presets)[0]
        details = stock_details[symbol][first_preset]
        sector = details.get('sector', 'N/A')[:18]
        market_cap = details.get('market_cap', 0)
        
        # Format market cap
        if market_cap and market_cap >= 1e9:
            mkt_cap_str = f"${market_cap/1e9:.1f}B"
        elif market_cap and market_cap >= 1e6:
            mkt_cap_str = f"${market_cap/1e6:.1f}M"
        else:
            mkt_cap_str = "N/A"
        
        presets_str = ", ".join(sorted(presets))[:38]
        
        print(f"{symbol:<8} {preset_count:<3} {avg_score:<10.1f} {presets_str:<40} {sector:<20} {mkt_cap_str:<12}")

def create_detailed_analysis(overlapping_stocks, stock_details, output_dir, base_filename="overlap_analysis.tsv"):
    """Create detailed CSV analysis with all metrics for overlapping stocks"""
    if not overlapping_stocks:
        print("No overlapping stocks to analyze")
        return
    
    output_file = os.path.join(output_dir, base_filename)
    detailed_data = []
    
    for symbol, presets in overlapping_stocks.items():
        # Base info
        row = {
            'symbol': symbol,
            'preset_count': len(presets),
            'presets': '; '.join(sorted(presets))
        }
        
        # Initialize all preset columns with None
        all_preset_names = load_all_preset_names()
        for preset_name in all_preset_names:
            row[f'{preset_name}_score'] = None
            row[f'{preset_name}_rank'] = None
        
        # Collect metrics from presets this stock appears in
        all_scores = []
        all_ranks = []
        
        for preset in presets:
            details = stock_details[symbol][preset]
            all_scores.append(details['score'])
            all_ranks.append(details['rank'])
            
            # Add preset-specific data
            row[f'{preset}_score'] = details['score']
            row[f'{preset}_rank'] = details['rank']
        
        # Add aggregate metrics
        row['avg_score'] = sum(all_scores) / len(all_scores)
        row['avg_rank'] = sum(all_ranks) / len(all_ranks)
        row['best_rank'] = min(all_ranks)
        
        # Add fundamental data (from first preset)
        first_preset = list(presets)[0]
        fundamentals = stock_details[symbol][first_preset]
        
        for metric in ['pe_ratio', 'pb_ratio', 'roe', 'market_cap', 'sector', 
                      'return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y', 'return_5y',
                      'revenue_growth', 'earnings_growth', 'profit_margin', 'dividend_yield',
                      'volatility', 'beta', 'current_ratio', 'debt_to_equity']:
            row[metric] = fundamentals.get(metric)
        
        detailed_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(detailed_data)
    df = df.sort_values(['preset_count', 'avg_score'], ascending=[False, False])
    df.to_csv(output_file, index=False, sep='\t')
    
    print(f"\nDetailed analysis saved to: {output_file}")
    return df

def suggest_quality_preset_fix():
    """Suggest improved quality preset configuration"""
    print("\n=== SUGGESTED QUALITY PRESET FIX ===")
    
    improved_config = {
        "quality_stocks_relaxed": {
            "description": "High-quality stocks with relaxed but meaningful criteria",
            "filters": [
                {"field": "roe", "operator": "gt", "value": 0.10},  # 10% instead of 15%
                {"field": "debt_to_equity", "operator": "lt", "value": 1.0},  # 100% instead of 50%
                {"field": "current_ratio", "operator": "gt", "value": 0.8},  # 0.8 instead of 1.0
                {"field": "profit_margin", "operator": "gt", "value": 0.05},  # 5% instead of 10%
                {"field": "market_cap", "operator": "gt", "value": 1000000000}  # $1B instead of $5B
            ],
            "ranking": [
                {"field": "roe", "weight": 2.0, "direction": "higher"},
                {"field": "profit_margin", "weight": 1.5, "direction": "higher"},
                {"field": "current_ratio", "weight": 1.0, "direction": "higher"},
                {"field": "debt_to_equity", "weight": 1.0, "direction": "lower"}
            ],
            "top_n": 20
        }
    }
    
    print("Improved quality preset configuration:")
    print(json.dumps(improved_config, indent=2))
    
    # Save to file
    config_file = "screener/improved_quality_preset.json"
    with open(config_file, 'w') as f:
        json.dump(improved_config, f, indent=2)
    
    print(f"\nSaved improved configuration to: {config_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze stock screening preset overlaps with auto-detection and screening')
    parser.add_argument('--min-presets', type=int, default=2, 
                       help='Minimum number of presets a stock must appear in (default: 2)')
    parser.add_argument('--data-file', default='stock_data.json',
                       help='Data file to use for screening')
    parser.add_argument('--analyze-quality', action='store_true',
                       help='Analyze why quality preset returns no results')
    parser.add_argument('--skip-auto-screening', action='store_true',
                       help='Skip automatic screening of missing presets')
    parser.add_argument('--no-regenerate', action='store_true',
                       help='Do not force regenerate temp results (use existing if available)')
    
    args = parser.parse_args()
    
    print("=== INTELLIGENT STOCK SCREENING OVERLAP ANALYSIS ===")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create organized output directory
    output_dir = create_analysis_output_dir()
    print(f"Analysis output directory: {output_dir}")
    
    # Load all preset names from configuration
    all_presets = load_all_preset_names()
    print(f"\nConfigured presets: {', '.join(all_presets)}")
    
    # Check for missing preset results (force regenerate by default)
    force_regenerate = not args.no_regenerate
    missing_presets, existing_presets = check_missing_presets(all_presets, force_regenerate=force_regenerate)
    
    if missing_presets:
        if force_regenerate:
            print(f"\nRegenerating all {len(missing_presets)} preset results...")
        else:
            print(f"\nMissing preset results for today: {', '.join(missing_presets)}")
            print(f"Existing preset results: {', '.join(existing_presets) if existing_presets else 'None'}")
        
        if not args.skip_auto_screening:
            # Automatically run missing screenings
            success = run_missing_screenings(missing_presets, args.data_file)
            if not success:
                print("\n⚠️  Some preset screenings failed. Continuing with available data...")
        else:
            print("⚠️  Skipping auto-screening due to --skip-auto-screening flag")
    else:
        print(f"\n✓ All {len(all_presets)} preset results are up to date!")
    
    # Load results after potential auto-screening
    results = load_screening_results()
    
    if not results:
        print("\n❌ No screening results found. Please ensure data is available:")
        print(f"   python main.py --action fetch --sp500")
        print(f"   python main.py --action screen --preset value_stocks")
        return
    
    print(f"\n📊 Loaded results from {len(results)} presets: {list(results.keys())}")
    
    # Analyze quality preset issue if requested
    if args.analyze_quality or 'quality_stocks' not in results:
        analyze_quality_preset_issue(results)
        suggest_quality_preset_fix()
    
    # Find overlapping stocks
    overlapping_stocks, stock_details = find_overlapping_stocks(results, args.min_presets)
    
    # Create reports
    create_overlap_report(overlapping_stocks, stock_details, args.min_presets)
    
    # Create detailed analysis with organized output
    if overlapping_stocks:
        df = create_detailed_analysis(overlapping_stocks, stock_details, output_dir)
        
        # Also create a summary report
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== STOCK SCREENING OVERLAP ANALYSIS SUMMARY ===\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Presets analyzed: {', '.join(results.keys())}\n")
            f.write(f"Total unique stocks across all presets: {sum(len(df_result) for df_result in results.values())}\n")
            f.write(f"Stocks appearing in {args.min_presets}+ presets: {len(overlapping_stocks)}\n\n")
            
            if len(overlapping_stocks) > 0:
                max_presets = max(len(presets) for presets in overlapping_stocks.values())
                f.write(f"Maximum preset overlap: {max_presets}\n\n")
                
                f.write("=== TOP 10 CANDIDATES (by preset count and avg score) ===\n")
                top_candidates = df.head(10)
                for _, row in top_candidates.iterrows():
                    f.write(f"{row['symbol']:<6} - {row['preset_count']} presets, avg score: {row['avg_score']:.1f}, sector: {row['sector']}\n")
        
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total unique stocks across all presets: {sum(len(df_result) for df_result in results.values())}")
        print(f"Stocks appearing in {args.min_presets}+ presets: {len(overlapping_stocks)}")
        
        if len(overlapping_stocks) > 0:
            max_presets = max(len(presets) for presets in overlapping_stocks.values())
            print(f"Maximum preset overlap: {max_presets}")
            
            # Show top candidates
            print(f"\n=== TOP 10 CANDIDATES (by preset count and avg score) ===")
            top_candidates = df.head(10)
            for _, row in top_candidates.iterrows():
                print(f"{row['symbol']:<6} - {row['preset_count']} presets, avg score: {row['avg_score']:.1f}, sector: {row['sector']}")
        
        print(f"\n📁 All analysis files saved to: {output_dir}")
        print(f"   - overlap_analysis.tsv: Detailed overlap analysis")
        print(f"   - analysis_summary.txt: Summary report")
    
    else:
        print(f"\n⚠️  No stocks found appearing in {args.min_presets}+ presets")
        # Still create a summary file
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== STOCK SCREENING OVERLAP ANALYSIS SUMMARY ===\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Presets analyzed: {', '.join(results.keys())}\n")
            f.write(f"Result: No stocks found appearing in {args.min_presets}+ presets\n")

if __name__ == "__main__":
    main()