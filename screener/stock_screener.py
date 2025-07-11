"""
Stock screener with configurable filtering and ranking criteria.
Focused on fundamental analysis and actionable stock lists.
"""
import json
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
import os

class StockScreener:
    def __init__(self, data_dir: str = "data", results_dir: str = "analysis/temp_results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.data = None
        self.screened_stocks = []
        os.makedirs(results_dir, exist_ok=True)
        
    def load_data(self, filename: str) -> bool:
        """Load stock data from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Data file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded data for {len(self.data)} stocks")
        return True
    
    def get_fundamentals_df(self) -> pd.DataFrame:
        """Convert fundamentals data to DataFrame for analysis"""
        if not self.data:
            return pd.DataFrame()
        
        fundamentals_list = []
        for symbol, symbol_data in self.data.items():
            if 'fundamentals' in symbol_data:
                fundamentals_list.append(symbol_data['fundamentals'])
        
        df = pd.DataFrame(fundamentals_list)
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
    
    def get_preset_screens(self) -> Dict[str, Dict]:
        """Get predefined screening configurations from config file"""
        config_path = os.path.join(os.path.dirname(__file__), 'screening_presets.json')
        
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