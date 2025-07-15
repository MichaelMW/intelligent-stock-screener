"""
Stock screener with configurable filtering and ranking criteria.
Focused on fundamental analysis and actionable stock lists.
"""
import json
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_storage import DataStorage

class StockScreener:
    def __init__(self, data_dir: str = "data", results_dir: str = "analysis/temp_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.data = None
        self.screened_stocks = []
        self.storage = DataStorage(data_dir)
        os.makedirs(results_dir, exist_ok=True)
        
    def load_data(self, filename: str = None) -> bool:
        """Load stock data from SQLite database (with JSON fallback)"""
        try:
            # Try loading from SQLite database first
            symbols = self.storage.get_symbols()
            if symbols:
                print(f"Loading data from SQLite database: {len(symbols)} symbols found")
                self.data = {}
                
                # Load fundamentals for all symbols
                fundamentals_data = self.storage.get_fundamentals()
                
                for symbol in symbols:
                    if symbol in fundamentals_data:
                        # Note: We don't load price data for screening as it's not needed
                        # and would be memory intensive
                        self.data[symbol] = {
                            'fundamentals': fundamentals_data[symbol],
                            'price_data': None  # Not loaded for screening efficiency
                        }
                
                print(f"Loaded data for {len(self.data)} stocks from database")
                return True
        except Exception as e:
            print(f"Error loading from database: {e}")
        
        # Fallback to JSON file if database fails and filename provided
        if filename:
            print(f"Falling back to JSON file: {filename}")
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Data file not found: {filepath}")
                return False
            
            try:
                with open(filepath, 'r') as f:
                    self.data = json.load(f)
                
                print(f"Loaded data for {len(self.data)} stocks from JSON")
                return True
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                return False
        
        print("No data source available")
        return False
    
    def get_fundamentals_df(self) -> pd.DataFrame:
        """Convert fundamentals data to DataFrame for analysis"""
        if not self.data:
            # Try loading directly from database if no data loaded
            fundamentals_data = self.storage.get_fundamentals()
            if fundamentals_data:
                df = pd.DataFrame(list(fundamentals_data.values()))
                df = self._convert_numeric_columns(df)
                return df
            return pd.DataFrame()
        
        fundamentals_list = []
        for symbol, symbol_data in self.data.items():
            if 'fundamentals' in symbol_data:
                fundamentals_list.append(symbol_data['fundamentals'])
        
        df = pd.DataFrame(fundamentals_list)
        df = self._convert_numeric_columns(df)
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns from strings to proper numeric types"""
        if df.empty:
            return df
        
        # List of columns that should be numeric
        numeric_columns = [
            'market_cap', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'debt_to_equity',
            'current_ratio', 'quick_ratio', 'roe', 'roa', 'profit_margin',
            'revenue_growth', 'earnings_growth', 'dividend_yield', 'payout_ratio',
            'return_1y', 'return_3y', 'return_5y', 'volatility_1y', 'beta',
            'price_52w_high', 'price_52w_low', 'price_from_high', 'price_from_low',
            'current_price', 'free_cash_flow', 'enterprise_value'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def apply_filters(self, filters: List[Dict]) -> List[str]:
        """Apply screening filters to data"""
        if not self.data:
            print("No data loaded")
            return []
        
        df = self.get_fundamentals_df()
        if df.empty:
            print("No fundamentals data available")
            return []
        
        # Start with all symbols
        filtered_symbols = set(df['symbol'].tolist())
        
        for filter_config in filters:
            field = filter_config['field']
            operator = filter_config['operator']
            value = filter_config.get('value', None)
            
            if field not in df.columns:
                print(f"Warning: Field '{field}' not found in data")
                continue
            
            # Apply filter
            if operator == 'gt':
                mask = df[field] > value
            elif operator == 'lt':
                mask = df[field] < value
            elif operator == 'gte':
                mask = df[field] >= value
            elif operator == 'lte':
                mask = df[field] <= value
            elif operator == 'eq':
                mask = df[field] == value
            elif operator == 'ne':
                mask = df[field] != value
            elif operator == 'between':
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    mask = (df[field] >= value[0]) & (df[field] <= value[1])
                else:
                    print(f"Warning: 'between' operator requires list/tuple of 2 values")
                    continue
            elif operator == 'notnull':
                mask = df[field].notna()
            else:
                print(f"Warning: Unknown operator '{operator}'")
                continue
            
            # Apply mask and get symbols
            filtered_df = df[mask]
            current_symbols = set(filtered_df['symbol'].tolist())
            
            # Intersect with existing filtered symbols
            filtered_symbols = filtered_symbols.intersection(current_symbols)
            
            print(f"Filter {field} {operator} {value}: {len(filtered_symbols)} stocks remaining")
        
        return list(filtered_symbols)
    
    def rank_stocks(self, symbols: List[str], ranking_criteria: List[Dict]) -> List[Tuple[str, float]]:
        """Rank stocks based on multiple criteria"""
        if not symbols:
            return []
        
        df = self.get_fundamentals_df()
        df_filtered = df[df['symbol'].isin(symbols)].copy()
        
        if df_filtered.empty:
            return []
        
        # Initialize total score
        df_filtered['total_score'] = 0.0
        
        for criteria in ranking_criteria:
            field = criteria['field']
            weight = criteria.get('weight', 1.0)
            direction = criteria.get('direction', 'higher')  # 'higher' or 'lower'
            
            if field not in df_filtered.columns:
                print(f"Warning: Ranking field '{field}' not found in data")
                continue
            
            # Remove rows with null values for this field
            field_data = df_filtered[field].dropna()
            if field_data.empty:
                continue
            
            # Calculate percentile scores (0-100)
            if direction == 'higher':
                # Higher values get higher scores
                scores = field_data.rank(pct=True) * 100
            else:
                # Lower values get higher scores
                scores = (1 - field_data.rank(pct=True)) * 100
            
            # Apply weight and add to total score
            weighted_scores = scores * weight
            df_filtered.loc[scores.index, 'total_score'] += weighted_scores
            
            print(f"Ranking by {field} (weight: {weight}, direction: {direction})")
        
        # Sort by total score (descending)
        df_ranked = df_filtered.sort_values('total_score', ascending=False)
        
        # Return as list of tuples (symbol, score)
        ranked_stocks = [(row['symbol'], row['total_score']) for _, row in df_ranked.iterrows()]
        
        return ranked_stocks
    
    def screen_stocks(self, config: Dict) -> List[Dict]:
        """Run complete screening process"""
        if not self.data:
            print("No data loaded")
            return []
        
        # Apply filters
        filters = config.get('filters', [])
        filtered_symbols = self.apply_filters(filters)
        
        if not filtered_symbols:
            print("No stocks passed the filters")
            return []
        
        # Rank stocks
        ranking_criteria = config.get('ranking', [])
        ranked_stocks = self.rank_stocks(filtered_symbols, ranking_criteria)
        
        # Get top N stocks
        top_n = config.get('top_n', 10)
        top_stocks = ranked_stocks[:top_n]
        
        # Build results with detailed information
        results = []
        df = self.get_fundamentals_df()
        
        for symbol, score in top_stocks:
            stock_data = df[df['symbol'] == symbol].iloc[0].to_dict()
            stock_data['screening_score'] = score
            results.append(stock_data)
        
        self.screened_stocks = results
        return results
    
    def analyze_all_presets(self, top_n: int = 20) -> Dict:
        """Run all presets and analyze overlaps between them"""
        print("🔍 Running comprehensive preset analysis...")
        
        # Get all available presets
        presets = self.get_preset_screens()
        
        # Run each preset and collect results
        preset_results = {}
        all_stocks = set()
        
        for preset_name, preset_config in presets.items():
            print(f"\n--- Running {preset_name} preset ---")
            preset_config['top_n'] = top_n
            
            # Run the screening
            results = self.screen_stocks(preset_config)
            
            if results:
                # Extract symbols from results
                symbols = [stock['symbol'] for stock in results]
                preset_results[preset_name] = {
                    'symbols': symbols,
                    'description': preset_config.get('description', ''),
                    'count': len(symbols),
                    'results': results
                }
                all_stocks.update(symbols)
                print(f"✅ {preset_name}: {len(symbols)} stocks")
            else:
                preset_results[preset_name] = {
                    'symbols': [],
                    'description': preset_config.get('description', ''),
                    'count': 0,
                    'results': []
                }
                print(f"❌ {preset_name}: No results")
        
        # Calculate overlaps
        print(f"\n📊 Analyzing overlaps across {len(preset_results)} presets...")
        
        # Find stocks that appear in multiple presets
        symbol_counts = {}
        for symbol in all_stocks:
            count = sum(1 for preset_data in preset_results.values() if symbol in preset_data['symbols'])
            symbol_counts[symbol] = count
        
        # Group by overlap count
        overlap_groups = {}
        for symbol, count in symbol_counts.items():
            if count not in overlap_groups:
                overlap_groups[count] = []
            overlap_groups[count].append(symbol)
        
        # Find specific overlaps between preset pairs
        preset_pairs = {}
        preset_names = list(preset_results.keys())
        
        for i, preset1 in enumerate(preset_names):
            for j, preset2 in enumerate(preset_names[i+1:], i+1):
                set1 = set(preset_results[preset1]['symbols'])
                set2 = set(preset_results[preset2]['symbols'])
                overlap = set1.intersection(set2)
                if overlap:
                    preset_pairs[f"{preset1} & {preset2}"] = list(overlap)
        
        return {
            'preset_results': preset_results,
            'overlap_groups': overlap_groups,
            'preset_pairs': preset_pairs,
            'total_unique_stocks': len(all_stocks),
            'analysis_summary': {
                'total_presets': len(preset_results),
                'successful_presets': len([p for p in preset_results.values() if p['count'] > 0]),
                'total_picks': sum(p['count'] for p in preset_results.values()),
                'unique_stocks': len(all_stocks)
            }
        }
    
    def print_analysis_results(self, analysis_results: Dict):
        """Print comprehensive analysis results"""
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE PRESET ANALYSIS RESULTS")
        print("="*80)
        
        # Summary
        summary = analysis_results['analysis_summary']
        print(f"\n📊 SUMMARY:")
        print(f"Total presets analyzed: {summary['total_presets']}")
        print(f"Successful presets: {summary['successful_presets']}")
        print(f"Total stock picks: {summary['total_picks']}")
        print(f"Unique stocks found: {summary['unique_stocks']}")
        
        # Preset results
        print(f"\n🎯 PRESET RESULTS:")
        preset_results = analysis_results['preset_results']
        
        for preset_name, data in preset_results.items():
            print(f"\n{preset_name}: {data['count']} stocks")
            print(f"  Description: {data['description']}")
            if data['symbols']:
                print(f"  Top picks: {', '.join(data['symbols'][:5])}")
                if len(data['symbols']) > 5:
                    print(f"  ... and {len(data['symbols']) - 5} more")
        
        # Overlap analysis
        print(f"\n🔗 OVERLAP ANALYSIS:")
        overlap_groups = analysis_results['overlap_groups']
        
        # Show stocks by overlap count (highest first)
        for count in sorted(overlap_groups.keys(), reverse=True):
            if count > 1:
                symbols = overlap_groups[count]
                print(f"\n📌 Stocks in {count} presets ({len(symbols)} stocks):")
                for symbol in symbols:
                    # Show which presets this symbol appears in
                    appearing_presets = [name for name, data in preset_results.items() if symbol in data['symbols']]
                    print(f"  {symbol}: {', '.join(appearing_presets)}")
        
        # Preset pair overlaps
        print(f"\n🤝 PRESET PAIR OVERLAPS:")
        preset_pairs = analysis_results['preset_pairs']
        
        if preset_pairs:
            for pair_name, symbols in preset_pairs.items():
                print(f"\n{pair_name}: {len(symbols)} stocks")
                print(f"  Overlap: {', '.join(symbols)}")
        else:
            print("No significant overlaps between preset pairs found.")
        
        # High-conviction picks (stocks in multiple presets)
        high_conviction_symbols = []
        for count in sorted(overlap_groups.keys(), reverse=True):
            if count >= 3:  # 3 or more presets
                high_conviction_symbols.extend(overlap_groups[count])
        
        if high_conviction_symbols:
            print(f"\n🎯 HIGH-CONVICTION PICKS (3+ presets):")
            for symbol in high_conviction_symbols:
                count = sum(1 for preset_data in preset_results.values() if symbol in preset_data['symbols'])
                appearing_presets = [name for name, data in preset_results.items() if symbol in data['symbols']]
                print(f"  {symbol} ({count} presets): {', '.join(appearing_presets)}")
        else:
            print(f"\n🎯 HIGH-CONVICTION PICKS (3+ presets): None found")

    def get_preset_screens(self) -> Dict[str, Dict]:
        """Get predefined screening configurations from config file"""
        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,          # “..”
                "config",
                "screening_presets.json"
                )
            )        
        
        try:
            with open(config_path, 'r') as f:
                presets = json.load(f)
            return presets
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using fallback default presets")
            return self._get_default_presets()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            print("Using fallback default presets")
            return self._get_default_presets()
    
    def _get_default_presets(self) -> Dict[str, Dict]:
        """Fallback default presets if config file is not available"""
        return {
            'value_stocks': {
                'description': 'Value stocks with low P/E and P/B ratios',
                'filters': [
                    {'field': 'pe_ratio', 'operator': 'lt', 'value': 20},
                    {'field': 'pb_ratio', 'operator': 'lt', 'value': 3},
                    {'field': 'market_cap', 'operator': 'gt', 'value': 1e9},
                    {'field': 'pe_ratio', 'operator': 'notnull'},
                    {'field': 'pb_ratio', 'operator': 'notnull'},
                ],
                'ranking': [
                    {'field': 'pe_ratio', 'weight': 2.0, 'direction': 'lower'},
                    {'field': 'pb_ratio', 'weight': 1.5, 'direction': 'lower'},
                    {'field': 'roe', 'weight': 1.0, 'direction': 'higher'},
                    {'field': 'dividend_yield', 'weight': 0.5, 'direction': 'higher'},
                ],
                'top_n': 20
            }
        }
    
    def export_results(self, filename: str, format: str = 'tsv'):
        """Export screening results to analysis/temp_results directory"""
        if not self.screened_stocks:
            print("No screening results to export")
            return
        
        filepath = os.path.join(self.results_dir, filename)
        
        if format == 'tsv':
            df = pd.DataFrame(self.screened_stocks)
            df.to_csv(filepath, index=False, sep='\t')
            print(f"Results exported to {filepath}")
        elif format == 'csv':
            df = pd.DataFrame(self.screened_stocks)
            df.to_csv(filepath, index=False)
            print(f"Results exported to {filepath}")
        else:
            print(f"Unsupported format: {format}. Use 'tsv' or 'csv'")
    
    def print_results(self, top_n: int = 10):
        """Print screening results in a formatted table"""
        if not self.screened_stocks:
            print("No screening results available")
            return
        
        print(f"\n=== TOP {min(top_n, len(self.screened_stocks))} STOCKS ===")
        print("-" * 120)
        
        header = f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'P/E':<8} {'P/B':<8} {'ROE':<8} {'Mkt Cap':<12} {'Sector':<20}"
        print(header)
        print("-" * 120)
        
        for i, stock in enumerate(self.screened_stocks[:top_n]):
            rank = i + 1
            symbol = stock.get('symbol', 'N/A')
            score = f"{stock.get('screening_score', 0):.1f}"
            pe = f"{stock.get('pe_ratio', 'N/A'):.1f}" if stock.get('pe_ratio') else 'N/A'
            pb = f"{stock.get('pb_ratio', 'N/A'):.1f}" if stock.get('pb_ratio') else 'N/A'
            roe = f"{stock.get('roe', 'N/A'):.1%}" if stock.get('roe') else 'N/A'
            mkt_cap = f"${stock.get('market_cap', 0)/1e9:.1f}B" if stock.get('market_cap') else 'N/A'
            sector = stock.get('sector', 'N/A')[:18] if stock.get('sector') else 'N/A'
            
            row = f"{rank:<4} {symbol:<8} {score:<8} {pe:<8} {pb:<8} {roe:<8} {mkt_cap:<12} {sector:<20}"
            print(row)
        
        print("-" * 120)

    def export_overlap_analysis(self, overlap_data: List[Dict], filename: str):
        """Export overlap analysis to TSV file matching the original format"""
        if not overlap_data:
            print("No overlap data to export")
            return
        
        try:
            with open(filename, 'w') as f:
                # Write header
                header_fields = ['symbol', 'preset_count', 'presets']
                
                # Add score/rank columns for each preset
                if overlap_data:
                    first_stock = overlap_data[0]
                    presets = []
                    for key in first_stock.keys():
                        if key.endswith('_score') and not key.startswith('avg_'):
                            preset_name = key.replace('_score', '')
                            presets.append(preset_name)
                    
                    for preset in presets:
                        header_fields.extend([f'{preset}_score', f'{preset}_rank'])
                
                # Add summary fields
                header_fields.extend(['avg_score', 'avg_rank', 'best_rank'])
                
                # Add fundamental fields
                fundamental_fields = ['pe_ratio', 'pb_ratio', 'roe', 'market_cap', 'sector', 
                                    'return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y', 'return_5y',
                                    'revenue_growth', 'earnings_growth', 'profit_margin', 'dividend_yield',
                                    'volatility', 'beta', 'current_ratio', 'debt_to_equity']
                header_fields.extend(fundamental_fields)
                
                f.write('\t'.join(header_fields) + '\n')
                
                # Write data rows
                for stock in overlap_data:
                    if stock.get('preset_count', 0) >= 2:  # Only include stocks in 2+ presets
                        row_data = []
                        
                        # Basic fields
                        row_data.append(str(stock.get('symbol', '')))
                        row_data.append(str(stock.get('preset_count', 0)))
                        row_data.append('; '.join(stock.get('presets', [])))
                        
                        # Score/rank fields for each preset
                        for preset in presets:
                            score_key = f'{preset}_score'
                            rank_key = f'{preset}_rank'
                            row_data.append(str(stock.get(score_key, '')))
                            row_data.append(str(stock.get(rank_key, '')))
                        
                        # Summary fields
                        row_data.append(str(stock.get('avg_score', '')))
                        row_data.append(str(stock.get('avg_rank', '')))
                        row_data.append(str(stock.get('best_rank', '')))
                        
                        # Fundamental fields
                        for field in fundamental_fields:
                            value = stock.get(field, '')
                            if value == '' or value is None:
                                row_data.append('')
                            else:
                                row_data.append(str(value))
                        
                        f.write('\t'.join(row_data) + '\n')
            
            print(f"Overlap analysis exported to: {filename}")
            
        except Exception as e:
            print(f"Error exporting overlap analysis: {e}")

if __name__ == "__main__":
    # Test the screener
    screener = StockScreener()
    
    # Check if test data exists
    if not screener.load_data("test_data.json"):
        print("No test data found. Run data_fetcher.py first to generate test data.")
        exit(1)
    
    # Test with value stocks preset
    presets = screener.get_preset_screens()
    print("Available presets:", list(presets.keys()))
    
    # Run value stocks screen
    print("\n=== RUNNING VALUE STOCKS SCREEN ===")
    results = screener.screen_stocks(presets['value_stocks'])
    
    if results:
        screener.print_results(10)
        screener.export_results("value_stocks_results.csv", "csv")
    else:
        print("No stocks found matching the criteria")