"""
Data fetcher for standalone stock picker using yfinance.
Minimal dependencies, reliable data source.
"""
import json
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Install with: pip install yfinance")
    exit(1)

import pandas as pd

class DataFetcher:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols from Wikipedia"""
        try:
            print("Fetching S&P 500 symbols from Wikipedia...")
            # Use pandas to scrape S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url, header=0)
            
            # Find the right table (first table usually contains the list)
            sp500_table = tables[0]
            print(f"Table columns: {list(sp500_table.columns)}")
            
            # Try different possible column names for symbols
            symbol_column = None
            for col in sp500_table.columns:
                if 'symbol' in col.lower() or 'ticker' in col.lower():
                    symbol_column = col
                    break
            
            if symbol_column is None:
                # If no symbol column found, assume first column
                symbol_column = sp500_table.columns[0]
                print(f"Using first column as symbol column: {symbol_column}")
            
            symbols = sp500_table[symbol_column].tolist()
            
            # Filter out invalid symbols and clean them
            cleaned_symbols = []
            for symbol in symbols:
                if isinstance(symbol, str) and len(symbol) <= 6:
                    # Clean symbols (remove dots for BRK.B -> BRK-B format)
                    cleaned_symbol = symbol.replace('.', '-').upper().strip()
                    # Validate symbol format (letters, numbers, and hyphens only)
                    if cleaned_symbol and all(c.isalnum() or c == '-' for c in cleaned_symbol):
                        cleaned_symbols.append(cleaned_symbol)
            
            if len(cleaned_symbols) < 400:  # S&P 500 should have ~500 symbols
                print(f"Warning: Only found {len(cleaned_symbols)} valid symbols, using fallback")
                raise ValueError("Insufficient valid symbols found")
            
            print(f"Successfully fetched {len(cleaned_symbols)} S&P 500 symbols")
            return cleaned_symbols
            
        except Exception as e:
            print(f"Error fetching S&P 500 symbols: {e}")
            print("Using fallback symbol list...")
            # Expanded fallback list with more representative S&P 500 stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
                'V', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'ADBE', 'NFLX',
                'CRM', 'NKE', 'PFE', 'TMO', 'ABT', 'COP', 'LLY', 'COST', 'AVGO', 'XOM',
                'ACN', 'DHR', 'VZ', 'CMCSA', 'TXN', 'QCOM', 'MRK', 'NEE', 'HON', 'T',
                'PM', 'UPS', 'RTX', 'AMD', 'LIN', 'SBUX', 'LOW', 'IBM', 'MDT', 'CVX'
            ]
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get stock price data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            return hist
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'price': info.get('currentPrice'),
                'volume': info.get('volume'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'last_updated': datetime.now().isoformat()
            }
            
            return fundamentals
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], period: str = "1y") -> Dict[str, Dict]:
        """Fetch both price and fundamental data for multiple symbols"""
        results = {}
        
        for i, symbol in enumerate(symbols):
            print(f"Fetching data for {symbol} ({i+1}/{len(symbols)})")
            
            # Get price data
            price_data = self.get_stock_data(symbol, period)
            
            # Get fundamental data
            fundamental_data = self.get_fundamentals(symbol)
            
            if price_data is not None and fundamental_data is not None:
                # Calculate additional metrics from price data
                current_price = price_data['Close'].iloc[-1]
                price_52w_high = price_data['High'].max()
                price_52w_low = price_data['Low'].min()
                
                # Add price-based metrics to fundamentals
                fundamental_data.update({
                    'current_price': current_price,
                    'price_52w_high': price_52w_high,
                    'price_52w_low': price_52w_low,
                    'price_from_high': (current_price / price_52w_high - 1) * 100,
                    'price_from_low': (current_price / price_52w_low - 1) * 100,
                })
                
                # Calculate returns for various time periods
                if len(price_data) >= 21:  # 1 month
                    month_return = (current_price / price_data['Close'].iloc[-21] - 1) * 100
                    fundamental_data['return_1m'] = month_return
                
                if len(price_data) >= 63:  # 3 months
                    quarter_return = (current_price / price_data['Close'].iloc[-63] - 1) * 100
                    fundamental_data['return_3m'] = quarter_return
                
                if len(price_data) >= 126:  # 6 months
                    six_month_return = (current_price / price_data['Close'].iloc[-126] - 1) * 100
                    fundamental_data['return_6m'] = six_month_return
                
                if len(price_data) >= 252:  # 1 year
                    year_return = (current_price / price_data['Close'].iloc[-252] - 1) * 100
                    fundamental_data['return_1y'] = year_return
                
                if len(price_data) >= 756:  # 3 years (approximate)
                    three_year_return = (current_price / price_data['Close'].iloc[-756] - 1) * 100
                    fundamental_data['return_3y'] = three_year_return
                
                if len(price_data) >= 1260:  # 5 years (approximate)
                    five_year_return = (current_price / price_data['Close'].iloc[-1260] - 1) * 100
                    fundamental_data['return_5y'] = five_year_return
                
                # Calculate volatility (annualized)
                if len(price_data) >= 252:
                    daily_returns = price_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                    fundamental_data['volatility'] = volatility
                
                results[symbol] = {
                    'fundamentals': fundamental_data,
                    'price_data': price_data
                }
        
        return results
    
    def save_data(self, data: Dict, filename: str, sync: bool = True):
        """Save data to JSON file with optional intelligent syncing"""
        if sync:
            self.save_data_sync(data, filename)
        else:
            self._save_data_direct(data, filename)
    
    def save_data_sync(self, new_data: Dict, filename: str):
        """Intelligently sync new data with existing data, preserving more comprehensive datasets"""
        filepath = os.path.join(self.data_dir, filename)
        
        # Load existing data if it exists
        existing_data = self.load_data(filename)
        
        if existing_data is None:
            print(f"No existing data found, saving new data to {filepath}")
            self._save_data_direct(new_data, filename)
            return
        
        print(f"Existing data found with {len(existing_data)} symbols, syncing with {len(new_data)} new symbols...")
        
        # Merge data intelligently
        merged_data = {}
        
        # Start with existing data
        for symbol, symbol_data in existing_data.items():
            merged_data[symbol] = symbol_data
        
        # Add or update with new data
        updated_count = 0
        new_count = 0
        
        for symbol, new_symbol_data in new_data.items():
            if symbol in merged_data:
                # Compare data comprehensiveness for existing symbols
                existing_symbol_data = merged_data[symbol]
                
                # Check if new price data is more comprehensive
                should_update = self._should_update_symbol_data(existing_symbol_data, new_symbol_data)
                
                if should_update:
                    merged_data[symbol] = new_symbol_data
                    updated_count += 1
                else:
                    # Keep existing data but update fundamentals (which change more frequently)
                    merged_data[symbol]['fundamentals'] = new_symbol_data['fundamentals']
            else:
                # New symbol, add it
                merged_data[symbol] = new_symbol_data
                new_count += 1
        
        print(f"Sync summary: {new_count} new symbols, {updated_count} updated symbols, {len(merged_data)} total symbols")
        self._save_data_direct(merged_data, filename)
    
    def _should_update_symbol_data(self, existing_data: Dict, new_data: Dict) -> bool:
        """Determine if new symbol data is more comprehensive than existing data"""
        # Check if we have price data in both
        existing_price = existing_data.get('price_data')
        new_price = new_data.get('price_data')
        
        # Handle None cases first
        if existing_price is None and new_price is not None:
            return True  # New data has price data, existing doesn't
        
        if existing_price is not None and new_price is None:
            return False  # Existing data has price data, new doesn't
        
        if existing_price is None and new_price is None:
            return True  # Neither has price data, update fundamentals
        
        # Both have price data, compare comprehensiveness
        try:
            # Convert to DataFrames if they're dicts
            if isinstance(existing_price, dict):
                existing_df = pd.DataFrame(existing_price)
            else:
                existing_df = existing_price
                
            if isinstance(new_price, dict):
                new_df = pd.DataFrame(new_price)
            else:
                new_df = new_price
            
            # Compare number of data points
            existing_points = len(existing_df) if not existing_df.empty else 0
            new_points = len(new_df) if not new_df.empty else 0
            
            # Update if new data has significantly more points (at least 20% more)
            return new_points > existing_points * 1.2
            
        except Exception as e:
            print(f"Error comparing price data: {e}")
            return True  # If comparison fails, update to be safe
    
    def _save_data_direct(self, data: Dict, filename: str):
        """Save data directly to JSON file (original save_data logic)"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            # Convert pandas DataFrames to dict for JSON serialization
            serializable_data = {}
            for symbol, symbol_data in data.items():
                price_data_dict = None
                if 'price_data' in symbol_data and symbol_data['price_data'] is not None:
                    # Convert DataFrame to dict with string keys
                    df = symbol_data['price_data']
                    price_data_dict = {}
                    for col in df.columns:
                        price_data_dict[col] = {}
                        for idx, val in df[col].items():
                            price_data_dict[col][idx.strftime('%Y-%m-%d')] = float(val) if pd.notna(val) else None
                
                serializable_data[symbol] = {
                    'fundamentals': symbol_data['fundamentals'],
                    'price_data': price_data_dict
                }
            json.dump(serializable_data, f, indent=2, default=str)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> Optional[Dict]:
        """Load data from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert price data back to DataFrame
        for symbol in data:
            if data[symbol]['price_data']:
                data[symbol]['price_data'] = pd.DataFrame(data[symbol]['price_data'])
        
        return data
    
    def export_fundamentals_csv(self, data: Dict, filename: str):
        """Export fundamentals to CSV for analysis"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not data:
            print("No data to export")
            return
        
        # Get all fundamental keys from first symbol
        first_symbol = list(data.keys())[0]
        fundamental_keys = list(data[first_symbol]['fundamentals'].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fundamental_keys)
            writer.writeheader()
            
            for symbol, symbol_data in data.items():
                writer.writerow(symbol_data['fundamentals'])
        
        print(f"Fundamentals exported to {filepath}")

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    print("Testing data fetcher with sample symbols...")
    
    data = fetcher.fetch_batch_data(test_symbols, period="6mo")
    
    if data:
        fetcher.save_data(data, "test_data.json")
        fetcher.export_fundamentals_csv(data, "test_fundamentals.csv")
        print("Test completed successfully!")
    else:
        print("Test failed - no data retrieved")