"""
Performance tracking for retrospective analysis of stock picks.
Tracks how selected stocks performed over time vs benchmarks.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Install with: pip install yfinance")
    exit(1)

class PerformanceTracker:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.portfolio_data = {}
        self.benchmark_data = {}
        
    def create_portfolio(self, symbols: List[str], portfolio_name: str, 
                        start_date: str, weights: Optional[List[float]] = None):
        """Create a portfolio from a list of symbols"""
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)  # Equal weights
        
        if len(weights) != len(symbols):
            raise ValueError("Number of weights must match number of symbols")
        
        portfolio = {
            'name': portfolio_name,
            'symbols': symbols,
            'weights': weights,
            'start_date': start_date,
            'created_date': datetime.now().isoformat()
        }
        
        self.portfolio_data[portfolio_name] = portfolio
        return portfolio
    
    def fetch_performance_data(self, symbols: List[str], start_date: str, 
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch price data for performance tracking"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        performance_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate daily returns
                    hist['Daily_Return'] = hist['Close'].pct_change()
                    hist['Cumulative_Return'] = (1 + hist['Daily_Return']).cumprod() - 1
                    performance_data[symbol] = hist
                    print(f"Fetched data for {symbol}: {len(hist)} days")
                else:
                    print(f"No data found for {symbol}")
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return performance_data
    
    def calculate_portfolio_performance(self, portfolio_name: str, 
                                      end_date: Optional[str] = None) -> Optional[Dict]:
        """Calculate portfolio performance metrics"""
        if portfolio_name not in self.portfolio_data:
            print(f"Portfolio '{portfolio_name}' not found")
            return None
        
        portfolio = self.portfolio_data[portfolio_name]
        symbols = portfolio['symbols']
        weights = portfolio['weights']
        start_date = portfolio['start_date']
        
        # Fetch price data
        price_data = self.fetch_performance_data(symbols, start_date, end_date)
        
        if not price_data:
            print("No price data available")
            return None
        
        # Find common date range
        common_dates = None
        for symbol, data in price_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) == 0:
            print("No common dates found across all symbols")
            return None
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=common_dates)
        
        for symbol, weight in zip(symbols, weights):
            if symbol in price_data:
                symbol_data = price_data[symbol].loc[common_dates]
                portfolio_returns += weight * symbol_data['Daily_Return']
        
        # Calculate portfolio metrics
        portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        
        # Performance metrics
        total_return = portfolio_cumulative.iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_values = 1 + portfolio_cumulative
        peak = cumulative_values.cummax()
        drawdown = (cumulative_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        performance_metrics = {
            'portfolio_name': portfolio_name,
            'start_date': start_date,
            'end_date': common_dates[-1].strftime('%Y-%m-%d'),
            'total_days': len(common_dates),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'symbols': symbols,
            'weights': weights,
            'daily_returns': portfolio_returns.tolist(),
            'cumulative_returns': portfolio_cumulative.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in common_dates]
        }
        
        return performance_metrics
    
    def get_benchmark_performance(self, benchmark_symbol: str = '^GSPC', 
                                start_date: str = '2023-01-01', 
                                end_date: Optional[str] = None) -> Optional[Dict]:
        """Get benchmark performance (default: S&P 500)"""
        benchmark_data = self.fetch_performance_data([benchmark_symbol], start_date, end_date)
        
        if benchmark_symbol not in benchmark_data:
            print(f"Failed to fetch benchmark data for {benchmark_symbol}")
            return None
        
        data = benchmark_data[benchmark_symbol]
        returns = data['Daily_Return'].dropna()
        
        # Calculate benchmark metrics
        total_return = data['Cumulative_Return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_values = 1 + data['Cumulative_Return']
        peak = cumulative_values.cummax()
        drawdown = (cumulative_values - peak) / peak
        max_drawdown = drawdown.min()
        
        benchmark_metrics = {
            'benchmark_symbol': benchmark_symbol,
            'start_date': start_date,
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': returns.tolist(),
            'cumulative_returns': data['Cumulative_Return'].tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in data.index]
        }
        
        return benchmark_metrics
    
    def compare_to_benchmark(self, portfolio_performance: Dict, 
                           benchmark_performance: Dict) -> Dict:
        """Compare portfolio performance to benchmark"""
        portfolio_return = portfolio_performance['annualized_return']
        benchmark_return = benchmark_performance['annualized_return']
        
        alpha = portfolio_return - benchmark_return
        
        comparison = {
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'outperformance': alpha > 0,
            'portfolio_sharpe': portfolio_performance['sharpe_ratio'],
            'benchmark_sharpe': benchmark_performance['sharpe_ratio'],
            'portfolio_volatility': portfolio_performance['volatility'],
            'benchmark_volatility': benchmark_performance['volatility'],
            'portfolio_max_drawdown': portfolio_performance['max_drawdown'],
            'benchmark_max_drawdown': benchmark_performance['max_drawdown']
        }
        
        return comparison
    
    def backtest_screening_strategy(self, screening_results: List[Dict], 
                                  start_date: str, rebalance_frequency: str = 'quarterly',
                                  top_n: int = 10) -> Dict:
        """Backtest a screening strategy with periodic rebalancing"""
        # This would implement a full backtesting framework
        # For now, return a simplified version
        
        # Get top N stocks from screening results
        top_stocks = screening_results[:top_n]
        symbols = [stock['symbol'] for stock in top_stocks]
        
        # Create equal-weighted portfolio
        portfolio_name = f"screening_strategy_{datetime.now().strftime('%Y%m%d')}"
        portfolio = self.create_portfolio(symbols, portfolio_name, start_date)
        
        # Calculate performance
        performance = self.calculate_portfolio_performance(portfolio_name)
        
        return performance
    
    def save_performance_data(self, data: Dict, filename: str):
        """Save performance data to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Performance data saved to {filepath}")
    
    def print_performance_report(self, portfolio_performance: Dict, 
                               benchmark_performance: Optional[Dict] = None):
        """Print a formatted performance report"""
        print(f"\n=== PERFORMANCE REPORT: {portfolio_performance['portfolio_name']} ===")
        print(f"Period: {portfolio_performance['start_date']} to {portfolio_performance['end_date']}")
        print(f"Duration: {portfolio_performance['total_days']} days")
        print("-" * 60)
        
        print(f"Total Return:      {portfolio_performance['total_return']:.2%}")
        print(f"Annualized Return: {portfolio_performance['annualized_return']:.2%}")
        print(f"Volatility:        {portfolio_performance['volatility']:.2%}")
        print(f"Sharpe Ratio:      {portfolio_performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:      {portfolio_performance['max_drawdown']:.2%}")
        print(f"Win Rate:          {portfolio_performance['win_rate']:.2%}")
        
        if benchmark_performance:
            print("\n=== BENCHMARK COMPARISON ===")
            comparison = self.compare_to_benchmark(portfolio_performance, benchmark_performance)
            print(f"Portfolio Return:  {comparison['portfolio_return']:.2%}")
            print(f"Benchmark Return:  {comparison['benchmark_return']:.2%}")
            print(f"Alpha:             {comparison['alpha']:.2%}")
            print(f"Outperformance:    {'Yes' if comparison['outperformance'] else 'No'}")
        
        print("=" * 60)

if __name__ == "__main__":
    # Test the performance tracker
    tracker = PerformanceTracker()
    
    # Create a test portfolio
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    
    portfolio = tracker.create_portfolio(test_symbols, 'test_portfolio', start_date)
    
    # Calculate performance
    performance = tracker.calculate_portfolio_performance('test_portfolio')
    
    if performance:
        # Get benchmark performance
        benchmark = tracker.get_benchmark_performance('^GSPC', start_date)
        
        # Print report
        tracker.print_performance_report(performance, benchmark)
        
        # Save data
        tracker.save_performance_data(performance, 'test_portfolio_performance.json')
        
        if benchmark:
            tracker.save_performance_data(benchmark, 'benchmark_performance.json')
    else:
        print("Failed to calculate performance")