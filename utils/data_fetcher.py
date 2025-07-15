"""
Data fetcher for standalone stock picker using yfinance.
Minimal dependencies, reliable data source.
"""
import json
import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from data_storage import DataStorage

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
        self.storage = DataStorage(data_dir)
    
    def _calculate_date_based_return(self, price_data: pd.DataFrame, months_back: int, max_tolerance_days: int = 10) -> Optional[float]:
        """
        Calculate return based on actual date calculation rather than hardcoded day counts.
        
        Args:
            price_data: DataFrame with datetime index and Close column
            months_back: Number of months to look back
            max_tolerance_days: Maximum days tolerance when finding the target date
            
        Returns:
            Return percentage or None if calculation not possible
        """
        if price_data is None or len(price_data) < 10:
            return None
        
        # Get the latest date and current price
        latest_date = price_data.index.max()
        current_price = price_data['Close'].iloc[-1]
        
        # Calculate target date (approximate months back from latest date)
        # Use 30.44 days per month as average (365.25/12)
        days_back = int(months_back * 30.44)
        target_date = latest_date - timedelta(days=days_back)
        
        # Find the closest date within tolerance
        price_data_sorted = price_data.sort_index()
        
        # Find dates within tolerance of target date
        date_diff = abs(price_data_sorted.index - target_date)
        min_diff = date_diff.min()
        
        # Check if we have a date within tolerance
        if min_diff.days > max_tolerance_days:
            return None
        
        # Find the closest date
        closest_date_mask = date_diff == min_diff
        closest_date = price_data_sorted.index[closest_date_mask][0]
        historical_price = price_data_sorted.loc[closest_date, 'Close']
        
        # Calculate return
        return_pct = (current_price / historical_price - 1) * 100
        
        return return_pct
        
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
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'last_updated': datetime.now().isoformat()
            }
            
            return fundamentals
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None
    
    def _fetch_symbol_batch(self, symbol_batch: List[str], period: str, results_queue: queue.Queue) -> int:
        """Fetch data for a batch of symbols (multi-threaded worker function)"""
        fetched_count = 0
        
        for symbol in symbol_batch:
            try:
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
                    
                    # Calculate returns using date-based approach (more accurate)
                    return_periods = [
                        (1, 'return_1m'),     # 1 month
                        (3, 'return_3m'),     # 3 months
                        (6, 'return_6m'),     # 6 months
                        (12, 'return_1y'),    # 1 year
                        (36, 'return_3y'),    # 3 years
                        (60, 'return_5y'),    # 5 years
                        (120, 'return_10y'),  # 10 years
                    ]
                    
                    for months_back, return_key in return_periods:
                        return_value = self._calculate_date_based_return(price_data, months_back)
                        if return_value is not None:
                            fundamental_data[return_key] = return_value
                    
                    # Calculate volatility (annualized)
                    if len(price_data) >= 252:
                        daily_returns = price_data['Close'].pct_change().dropna()
                        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                        fundamental_data['volatility'] = volatility
                    
                    # Queue the result for database writing (thread-safe)
                    results_queue.put((symbol, fundamental_data, price_data))
                    fetched_count += 1
                    
                else:
                    # Still queue a failure result to maintain count
                    results_queue.put((symbol, None, None))
                    
            except Exception as e:
                # Queue failure result and continue
                results_queue.put((symbol, None, None))
                continue
        
        return fetched_count

    def _fetch_database_writer_thread(self, results_queue: queue.Queue, total_expected: int) -> int:
        """Single thread to handle all database writes during fetch (avoids locking issues)"""
        saved_count = 0
        received_count = 0
        results = {}
        
        # Create dedicated storage connection for writing with WAL mode
        storage = DataStorage(self.data_dir)
        
        # Enable WAL mode for better concurrency
        try:
            import sqlite3
            with sqlite3.connect(storage.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not optimize database settings: {e}")
        
        while received_count < total_expected:
            try:
                # Get result from queue (blocks until available)
                symbol, fundamental_data, price_data = results_queue.get(timeout=120)  # Longer timeout for network operations
                received_count += 1
                
                if fundamental_data is not None and price_data is not None:
                    # Save to database
                    if storage.save_stock_data(symbol, fundamental_data, price_data):
                        saved_count += 1
                        print(f"✓ Saved data for {symbol} to database ({saved_count}/{total_expected})")
                        
                        # Keep in results for compatibility
                        results[symbol] = {
                            'fundamentals': fundamental_data,
                            'price_data': price_data
                        }
                    else:
                        print(f"⚠ Failed to save {symbol} to database")
                else:
                    print(f"⚠ Failed to fetch complete data for {symbol}, skipping")
                
                # Mark task as done
                results_queue.task_done()
                
            except queue.Empty:
                # Timeout - likely no more data coming
                print(f"⏰ Fetch database writer timeout - processed {received_count}/{total_expected}")
                break
            except Exception as e:
                print(f"Fetch database writer error for item {received_count}: {e}")
                # Still mark as done to avoid blocking
                try:
                    results_queue.task_done()
                except:
                    pass
                continue
        
        # Store results for return
        self._fetch_results = results
        return saved_count

    def fetch_batch_data(self, symbols: List[str], period: str = "1y", save_incrementally: bool = True, max_workers: int = None) -> Dict[str, Dict]:
        """Fetch both price and fundamental data for multiple symbols using multi-threading and SQLite storage"""
        
        # Handle single-threaded fallback for small datasets
        if len(symbols) <= 5:
            print(f"📝 Using single-threaded mode for small dataset ({len(symbols)} symbols)")
            return self._fetch_batch_data_sequential(symbols, period)
        
        print(f"🌐 Fetching data for {len(symbols)} symbols using multi-threading...")
        
        # Determine optimal batch size and worker count
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(8, max(1, cpu_count - 1))  # Limit to 8 for network operations
        
        # Smaller batches for network operations to reduce timeout risk
        batch_size = max(5, len(symbols) // (max_workers * 4))
        
        # Split symbols into batches
        symbol_batches = [
            symbols[i:i + batch_size] 
            for i in range(0, len(symbols), batch_size)
        ]
        
        total_threads = max_workers + 1
        cpu_count = os.cpu_count() or 4
        print(f"🚀 Using {max_workers} fetch threads + 1 database writer = {total_threads} total threads")
        print(f"💻 CPU utilization: {total_threads}/{cpu_count} cores ({(total_threads/cpu_count)*100:.0f}%)")
        print(f"📦 {len(symbol_batches)} batches (batch size: ~{batch_size})")
        
        # Create thread-safe queue for database operations
        queue_size = min(200, len(symbols) + 50)  # Moderate queue size for network operations
        results_queue = queue.Queue(maxsize=queue_size)
        
        fetched_count = 0
        completed_batches = 0
        
        # Initialize results storage
        self._fetch_results = {}
        
        # Start database writer thread
        def db_writer():
            return self._fetch_database_writer_thread(results_queue, len(symbols))
        
        with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
            # Start database writer thread
            db_writer_future = executor.submit(db_writer)
            
            # Submit all fetch batch jobs
            future_to_batch = {
                executor.submit(self._fetch_symbol_batch, batch, period, results_queue): i
                for i, batch in enumerate(symbol_batches)
            }
            
            # Collect fetch results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_fetched = future.result()
                    fetched_count += batch_fetched
                    completed_batches += 1
                    
                    progress = completed_batches / len(symbol_batches) * 100
                    queue_size_current = results_queue.qsize()
                    print(f"Progress: {completed_batches}/{len(symbol_batches)} batches ({progress:.1f}%) - {fetched_count} fetched, queue: {queue_size_current}")
                    
                except Exception as e:
                    print(f"Fetch batch {batch_idx} failed: {e}")
                    continue
            
            # Wait for database writer to finish
            print("⏳ Waiting for database writes to complete...")
            saved_count = db_writer_future.result()
        
        print(f"\n📊 Batch summary: {saved_count}/{len(symbols)} symbols saved to database")
        return self._fetch_results
    
    def _fetch_batch_data_sequential(self, symbols: List[str], period: str) -> Dict[str, Dict]:
        """Sequential fallback for small datasets (≤5 symbols)"""
        results = {}
        successful_saves = 0
        
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
                
                # Calculate returns using date-based approach (more accurate)
                return_periods = [
                    (1, 'return_1m'),     # 1 month
                    (3, 'return_3m'),     # 3 months
                    (6, 'return_6m'),     # 6 months
                    (12, 'return_1y'),    # 1 year
                    (36, 'return_3y'),    # 3 years
                    (60, 'return_5y'),    # 5 years
                    (120, 'return_10y'),  # 10 years
                ]
                
                for months_back, return_key in return_periods:
                    return_value = self._calculate_date_based_return(price_data, months_back)
                    if return_value is not None:
                        fundamental_data[return_key] = return_value
                
                # Calculate volatility (annualized)
                if len(price_data) >= 252:
                    daily_returns = price_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                    fundamental_data['volatility'] = volatility
                
                # Save directly to SQLite database
                if self.storage.save_stock_data(symbol, fundamental_data, price_data):
                    successful_saves += 1
                    print(f"✓ Saved data for {symbol} to database")
                    
                    # Keep in results for compatibility
                    results[symbol] = {
                        'fundamentals': fundamental_data,
                        'price_data': price_data
                    }
                else:
                    print(f"⚠ Failed to save {symbol} to database")
            else:
                print(f"⚠ Failed to fetch complete data for {symbol}, skipping")
        
        print(f"\n📊 Sequential summary: {successful_saves}/{len(symbols)} symbols saved to database")
        return results
    
    def save_data(self, data: Dict, filename: str, sync: bool = True):
        """Save data to JSON file for backward compatibility"""
        print(f"Warning: save_data() with JSON is deprecated. Data is now stored in SQLite database.")
        print(f"Use export_to_json() to create JSON exports for compatibility.")
        
        # For backward compatibility, save to JSON
        self._save_data_direct(data, filename)
    
    def _save_data_direct(self, data: Dict, filename: str):
        """Save data directly to JSON file (for backward compatibility)"""
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
                            # Handle both datetime objects and string dates
                            if hasattr(idx, 'strftime'):
                                date_str = idx.strftime('%Y-%m-%d')
                            else:
                                date_str = str(idx)  # Already a string
                            price_data_dict[col][date_str] = float(val) if pd.notna(val) else None
                
                serializable_data[symbol] = {
                    'fundamentals': symbol_data['fundamentals'],
                    'price_data': price_data_dict
                }
            json.dump(serializable_data, f, indent=2, default=str)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> Optional[Dict]:
        """Load data from SQLite database (with JSON fallback for compatibility)"""
        # First try to load from SQLite database
        try:
            symbols = self.storage.get_symbols()
            if symbols:
                print(f"Loading data from SQLite database: {len(symbols)} symbols found")
                results = {}
                
                for symbol in symbols:
                    fundamentals = self.storage.get_fundamentals([symbol]).get(symbol, {})
                    price_data = self.storage.get_price_data(symbol)
                    
                    results[symbol] = {
                        'fundamentals': fundamentals,
                        'price_data': price_data
                    }
                
                return results
        except Exception as e:
            print(f"Error loading from SQLite database: {e}")
        
        # Fallback to JSON file if SQLite fails
        print(f"Falling back to JSON file: {filename}")
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Corrupted JSON file {filepath}: {e}")
            return None
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None
        
        # Convert price data back to DataFrame with proper datetime index
        for symbol in data:
            if data[symbol]['price_data']:
                df = pd.DataFrame(data[symbol]['price_data'])
                # Convert string dates back to datetime index
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass  # Keep original index if conversion fails
                data[symbol]['price_data'] = df
        
        return data
    
    def export_fundamentals_csv(self, data: Dict, filename: str):
        """Export fundamentals to CSV for analysis"""
        filepath = os.path.join(self.data_dir, filename)
        
        # If no data provided, export from database
        if not data:
            print("No data provided, exporting from database...")
            fundamentals_data = self.storage.get_fundamentals()
            if not fundamentals_data:
                print("No data to export")
                return
            
            # Define preferred column order
            preferred_order = [
                'symbol', 'market_cap', 'pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio',
                'debt_to_equity', 'roe', 'roa', 'current_ratio', 'quick_ratio', 'revenue_growth',
                'earnings_growth', 'profit_margin', 'operating_margin', 'dividend_yield', 'beta',
                'price', 'volume', 'week_52_high', 'week_52_low', 'sector', 'industry',
                'current_price', 'price_52w_high', 'price_52w_low', 'price_from_high', 'price_from_low',
                'return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y', 'return_5y', 'return_10y',
                'volatility', 'last_updated'
            ]
            
            # Get all available keys from first symbol
            first_symbol = list(fundamentals_data.keys())[0]
            available_keys = set(fundamentals_data[first_symbol].keys())
            
            # Build ordered fieldnames: preferred order first, then any additional keys
            fundamental_keys = []
            for key in preferred_order:
                if key in available_keys:
                    fundamental_keys.append(key)
                    available_keys.remove(key)
            
            # Add any remaining keys that weren't in preferred order
            fundamental_keys.extend(sorted(available_keys))
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fundamental_keys)
                writer.writeheader()
                
                for symbol, fund_data in fundamentals_data.items():
                    writer.writerow(fund_data)
        else:
            # Export from provided data (backward compatibility)
            first_symbol = list(data.keys())[0]
            
            # Use same preferred order for consistency
            preferred_order = [
                'symbol', 'market_cap', 'pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio',
                'debt_to_equity', 'roe', 'roa', 'current_ratio', 'quick_ratio', 'revenue_growth',
                'earnings_growth', 'profit_margin', 'operating_margin', 'dividend_yield', 'beta',
                'price', 'volume', 'week_52_high', 'week_52_low', 'sector', 'industry',
                'current_price', 'price_52w_high', 'price_52w_low', 'price_from_high', 'price_from_low',
                'return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y', 'return_5y', 'return_10y',
                'volatility', 'last_updated'
            ]
            
            available_keys = set(data[first_symbol]['fundamentals'].keys())
            
            # Build ordered fieldnames
            fundamental_keys = []
            for key in preferred_order:
                if key in available_keys:
                    fundamental_keys.append(key)
                    available_keys.remove(key)
            
            # Add any remaining keys
            fundamental_keys.extend(sorted(available_keys))
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fundamental_keys)
                writer.writeheader()
                
                for symbol, symbol_data in data.items():
                    writer.writerow(symbol_data['fundamentals'])
        
        print(f"Fundamentals exported to {filepath}")
    
    def export_to_json(self, filename: str = "stock_data.json"):
        """Export database data to JSON format for compatibility"""
        return self.storage.export_to_json(filename)
    
    def get_database_stats(self):
        """Get database statistics"""
        return self.storage.get_database_stats()
    
    def _regenerate_symbol_batch(self, symbol_batch: List[str], results_queue: queue.Queue) -> int:
        """Regenerate fundamentals for a batch of symbols (thread-safe) - computation only"""
        processed_count = 0
        
        # Create a read-only storage connection for this thread with optimization
        storage = DataStorage(self.data_dir)
        
        # Optimize for reading
        try:
            import sqlite3
            with sqlite3.connect(storage.db_path) as conn:
                conn.execute("PRAGMA cache_size=5000")
                conn.execute("PRAGMA temp_store=MEMORY")
        except Exception:
            pass  # Continue if optimization fails
        
        for symbol in symbol_batch:
            try:
                # Get existing fundamentals and price data (read-only operations)
                fundamentals = storage.get_fundamentals([symbol]).get(symbol, {})
                price_data = storage.get_price_data(symbol)
                
                if price_data is None or len(price_data) < 10:
                    continue
                
                # Recalculate returns with existing price data
                current_price = price_data['Close'].iloc[-1]
                
                # Recalculate all return periods using date-based approach
                return_periods = [
                    (1, 'return_1m'),     # 1 month
                    (3, 'return_3m'),     # 3 months
                    (6, 'return_6m'),     # 6 months
                    (12, 'return_1y'),    # 1 year
                    (36, 'return_3y'),    # 3 years
                    (60, 'return_5y'),    # 5 years
                    (120, 'return_10y'),  # 10 years
                ]
                
                for months_back, return_key in return_periods:
                    return_value = self._calculate_date_based_return(price_data, months_back)
                    if return_value is not None:
                        fundamentals[return_key] = return_value
                
                # Recalculate volatility
                if len(price_data) >= 252:
                    daily_returns = price_data['Close'].pct_change().dropna()
                    fundamentals['volatility'] = daily_returns.std() * (252 ** 0.5) * 100
                
                # Update timestamp
                fundamentals['last_updated'] = datetime.now().isoformat()
                
                # Queue the result for database writing (thread-safe)
                results_queue.put((symbol, fundamentals, price_data))
                processed_count += 1
                
            except Exception as e:
                # Silently continue for batch processing
                continue
        
        return processed_count
    
    def _database_writer_thread(self, results_queue: queue.Queue, total_expected: int) -> int:
        """Single thread to handle all database writes (avoids locking issues)"""
        updated_count = 0
        received_count = 0
        
        # Create dedicated storage connection for writing with WAL mode
        storage = DataStorage(self.data_dir)
        
        # Enable WAL mode for better concurrency (write-ahead logging)
        try:
            import sqlite3
            with sqlite3.connect(storage.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not optimize database settings: {e}")
        
        while received_count < total_expected:
            try:
                # Get result from queue (blocks until available)
                symbol, fundamentals, price_data = results_queue.get(timeout=60)  # Longer timeout
                received_count += 1
                
                # Update fundamentals in database (no new data fetching)
                if storage.update_fundamentals_only(symbol, fundamentals):
                    updated_count += 1
                    if updated_count % 50 == 0:  # Progress update every 50 updates
                        print(f"🔄 Updated {updated_count}/{total_expected} fundamentals")
                
                # Mark task as done
                results_queue.task_done()
                
            except queue.Empty:
                # Timeout - likely no more data coming
                print(f"⏰ Database writer timeout - processed {received_count}/{total_expected}")
                break
            except Exception as e:
                print(f"Database writer error for item {received_count}: {e}")
                # Still mark as done to avoid blocking
                try:
                    results_queue.task_done()
                except:
                    pass
                continue
        
        return updated_count

    def regenerate_fundamentals(self, max_workers: int = None):
        """Regenerate fundamentals from existing price data in database with thread-safe multi-threading"""
        print("🔄 Recalculating fundamentals from existing price data...")
        
        # Get all symbols from database
        symbols = self.storage.get_symbols()
        if not symbols:
            print("❌ No symbols found in database")
            return False
        
        print(f"📊 Processing {len(symbols)} symbols...")
        
        # Determine optimal batch size and worker count
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(1, cpu_count - 1)  # Use all CPUs except one for OS
        
        batch_size = max(10, len(symbols) // (max_workers * 2))  # Smaller batches for better progress tracking
        
        # Split symbols into batches
        symbol_batches = [
            symbols[i:i + batch_size] 
            for i in range(0, len(symbols), batch_size)
        ]
        
        total_threads = max_workers + 1
        cpu_count = os.cpu_count() or 4
        print(f"🚀 Using {max_workers} worker threads + 1 database writer = {total_threads} total threads")
        print(f"💻 CPU utilization: {total_threads}/{cpu_count} cores ({(total_threads/cpu_count)*100:.0f}%)")
        print(f"📦 {len(symbol_batches)} batches (batch size: ~{batch_size})")
        
        # Create thread-safe queue for database operations - larger queue to avoid blocking
        queue_size = min(500, len(symbols) + 100)  # Dynamic queue size based on symbol count
        results_queue = queue.Queue(maxsize=queue_size)
        
        processed_count = 0
        completed_batches = 0
        
        # Start database writer thread
        def db_writer():
            return self._database_writer_thread(results_queue, len(symbols))
        
        with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
            # Start database writer thread
            db_writer_future = executor.submit(db_writer)
            
            # Submit all computation batch jobs
            future_to_batch = {
                executor.submit(self._regenerate_symbol_batch, batch, results_queue): i
                for i, batch in enumerate(symbol_batches)
            }
            
            # Collect computation results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_processed = future.result()
                    processed_count += batch_processed
                    completed_batches += 1
                    
                    progress = completed_batches / len(symbol_batches) * 100
                    queue_size = results_queue.qsize()
                    print(f"Progress: {completed_batches}/{len(symbol_batches)} batches ({progress:.1f}%) - {processed_count} processed, queue: {queue_size}")
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    continue
            
            # Wait for database writer to finish
            print("⏳ Waiting for database writes to complete...")
            updated_count = db_writer_future.result()
        
        print(f"✅ Recalculated fundamentals for {updated_count}/{len(symbols)} symbols (no new data fetched)")
        return True

if __name__ == "__main__":
    # Test the data fetcher with SQLite
    fetcher = DataFetcher()
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    print("Testing data fetcher with sample symbols...")
    
    data = fetcher.fetch_batch_data(test_symbols, period="6mo")
    
    if data:
        print(f"Fetched data for {len(data)} symbols")
        
        # Test database stats
        stats = fetcher.get_database_stats()
        print(f"Database stats: {stats}")
        
        # Test CSV export
        fetcher.export_fundamentals_csv(None, "test_fundamentals.csv")
        
        # Test JSON export for compatibility
        fetcher.export_to_json("test_data.json")
        
        print("Test completed successfully!")
    else:
        print("Test failed - no data retrieved")