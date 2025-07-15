"""
SQLite-based data storage for stock screener.
Efficient storage and retrieval of stock fundamentals and price data.
"""
import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os


class DataStorage:
    """SQLite-based storage for stock data with efficient sync operations."""
    
    def __init__(self, data_dir: str = "data", db_name: str = "stock_data.db"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, db_name)
        os.makedirs(data_dir, exist_ok=True)
        self._init_database()
    
    def database_exists(self) -> bool:
        """Check if the database file exists and has data"""
        if not os.path.exists(self.db_path):
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fundamentals")
                count = cursor.fetchone()[0]
                return count > 0
        except Exception:
            return False
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create fundamentals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamentals (
                    symbol TEXT PRIMARY KEY,
                    market_cap REAL,
                    pe_ratio REAL,
                    forward_pe REAL,
                    pb_ratio REAL,
                    ps_ratio REAL,
                    peg_ratio REAL,
                    debt_to_equity REAL,
                    roe REAL,
                    roa REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    revenue_growth REAL,
                    earnings_growth REAL,
                    profit_margin REAL,
                    operating_margin REAL,
                    dividend_yield REAL,
                    beta REAL,
                    price REAL,
                    volume INTEGER,
                    week_52_high REAL,
                    week_52_low REAL,
                    sector TEXT,
                    industry TEXT,
                    current_price REAL,
                    price_52w_high REAL,
                    price_52w_low REAL,
                    price_from_high REAL,
                    price_from_low REAL,
                    return_1m REAL,
                    return_3m REAL,
                    return_6m REAL,
                    return_1y REAL,
                    return_3y REAL,
                    return_5y REAL,
                    return_10y REAL,
                    volatility REAL,
                    last_updated TEXT
                )
            ''')
            
            # Create price_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON fundamentals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date)')
            
            conn.commit()
    
    def save_stock_data(self, symbol: str, fundamentals: Dict[str, Any], price_data: pd.DataFrame) -> bool:
        """Save stock data (fundamentals + price data) for a single symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save fundamentals
                self._save_fundamentals(cursor, symbol, fundamentals)
                
                # Save price data
                if price_data is not None and not price_data.empty:
                    self._save_price_data(cursor, symbol, price_data)
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False
    
    def update_fundamentals_only(self, symbol: str, fundamentals: Dict[str, Any]) -> bool:
        """Update only fundamentals data (for regeneration from existing price data)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update only fundamentals
                self._save_fundamentals(cursor, symbol, fundamentals)
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating fundamentals for {symbol}: {e}")
            return False
    
    def _save_fundamentals(self, cursor: sqlite3.Cursor, symbol: str, fundamentals: Dict[str, Any]):
        """Save fundamentals data using INSERT OR REPLACE."""
        # Prepare fundamentals data with current timestamp
        fund_data = fundamentals.copy()
        fund_data['last_updated'] = datetime.now().isoformat()
        
        # Get column names and values
        columns = list(fund_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = [fund_data.get(col) for col in columns]
        
        query = f'''
            INSERT OR REPLACE INTO fundamentals ({', '.join(columns)})
            VALUES ({placeholders})
        '''
        
        cursor.execute(query, values)
    
    def _save_price_data(self, cursor: sqlite3.Cursor, symbol: str, price_data: pd.DataFrame):
        """Save price data using INSERT OR REPLACE."""
        # Prepare price data for insertion
        price_records = []
        for date, row in price_data.iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            price_records.append((
                symbol,
                date_str,
                float(row['Open']) if pd.notna(row['Open']) else None,
                float(row['High']) if pd.notna(row['High']) else None,
                float(row['Low']) if pd.notna(row['Low']) else None,
                float(row['Close']) if pd.notna(row['Close']) else None,
                int(row['Volume']) if pd.notna(row['Volume']) else None
            ))
        
        # Batch insert price data
        cursor.executemany('''
            INSERT OR REPLACE INTO price_data (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', price_records)
    
    def get_fundamentals(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get fundamentals data for specified symbols or all symbols."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            if symbols:
                placeholders = ', '.join(['?' for _ in symbols])
                query = f'SELECT * FROM fundamentals WHERE symbol IN ({placeholders})'
                cursor.execute(query, symbols)
            else:
                cursor.execute('SELECT * FROM fundamentals')
            
            results = {}
            for row in cursor.fetchall():
                symbol = row['symbol']
                # Convert row to dict, handling None values
                fund_data = {col: row[col] for col in row.keys()}
                results[symbol] = fund_data
            
            return results
    
    def get_price_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get price data for a specific symbol within date range."""
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT date, open, high, low, close, volume FROM price_data WHERE symbol = ?'
            params = [symbol]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY date'
            
            try:
                df = pd.read_sql_query(query, conn, params=params)
                if df.empty:
                    return None
                
                # Convert date column to datetime and set as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Rename columns to match original format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                return df
                
            except Exception as e:
                print(f"Error getting price data for {symbol}: {e}")
                return None
    
    def get_symbols(self) -> List[str]:
        """Get all available symbols in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT symbol FROM fundamentals ORDER BY symbol')
            return [row[0] for row in cursor.fetchall()]
    
    def symbol_exists(self, symbol: str) -> bool:
        """Check if a symbol exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM fundamentals WHERE symbol = ? LIMIT 1', (symbol,))
            return cursor.fetchone() is not None
    
    def get_last_updated(self, symbol: str) -> Optional[str]:
        """Get the last updated timestamp for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT last_updated FROM fundamentals WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_price_data_range(self, symbol: str) -> Optional[Tuple[str, str]]:
        """Get the date range of available price data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MIN(date), MAX(date) FROM price_data WHERE symbol = ?
            ''', (symbol,))
            result = cursor.fetchone()
            return (result[0], result[1]) if result and result[0] else None
    
    def delete_symbol(self, symbol: str) -> bool:
        """Delete all data for a specific symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM fundamentals WHERE symbol = ?', (symbol,))
                cursor.execute('DELETE FROM price_data WHERE symbol = ?', (symbol,))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting data for {symbol}: {e}")
            return False
    
    def export_to_json(self, filename: str) -> bool:
        """Export all data to JSON format for backward compatibility."""
        try:
            symbols = self.get_symbols()
            export_data = {}
            
            for symbol in symbols:
                fundamentals = self.get_fundamentals([symbol]).get(symbol, {})
                price_data = self.get_price_data(symbol)
                
                # Convert price data to the original JSON format
                price_data_dict = None
                if price_data is not None:
                    price_data_dict = {}
                    for col in price_data.columns:
                        price_data_dict[col] = {}
                        for date, value in price_data[col].items():
                            date_str = date.strftime('%Y-%m-%d')
                            price_data_dict[col][date_str] = float(value) if pd.notna(value) else None
                
                export_data[symbol] = {
                    'fundamentals': fundamentals,
                    'price_data': price_data_dict
                }
            
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Data exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get symbol count
            cursor.execute('SELECT COUNT(*) FROM fundamentals')
            symbol_count = cursor.fetchone()[0]
            
            # Get price data count
            cursor.execute('SELECT COUNT(*) FROM price_data')
            price_records = cursor.fetchone()[0]
            
            # Get database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'symbols': symbol_count,
                'price_records': price_records,
                'db_size_mb': db_size / (1024 * 1024),
                'db_path': self.db_path
            }


if __name__ == "__main__":
    # Test the DataStorage class
    storage = DataStorage()
    
    # Test basic functionality
    print("Database initialized successfully")
    stats = storage.get_database_stats()
    print(f"Database stats: {stats}")