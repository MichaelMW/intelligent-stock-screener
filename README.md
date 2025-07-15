# Intelligent Stock Screener

A comprehensive stock screening system with real-time data, multiple strategies, and advanced analysis capabilities.

## Features

- **Real Data**: Uses yfinance for current stock and fundamental data
- **9 Screening Strategies**: Value, growth, quality, momentum, long-term compounders, consistent performers, low-volatility growth, quality dividend stocks, and turnaround candidates
- **Multi-Preset Analysis**: Built-in analysis action identifies stocks that score well across multiple strategies
- **Performance Tracking**: Backtesting with benchmark comparison and risk metrics
- **Extended Timeframes**: 1M, 3M, 6M, 1Y, 3Y, 5Y return calculations with volatility metrics
- **Multi-threading Support**: Parallel data processing for improved performance
- **Database Storage**: SQLite backend with efficient data management and fundamentals recalculation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline with S&P 500 data (5-year period for comprehensive analysis)
python main.py --action full --sp500 --preset deep_value --top-n 20 --period 5y

# Multi-preset overlap analysis (new built-in action)
python main.py --action analyze --top-n 15
```

## Usage

### Core Actions
```bash
# Fetch data only (5-year period recommended for comprehensive long-term analysis)
python main.py --action fetch --sp500 --period 5y --workers 8

# Screen stocks only (requires existing data)
python main.py --action screen --preset high_growth --top-n 15

# Backtest performance only
python main.py --action backtest --preset fortress_balance_sheet

# Recalculate fundamentals from existing price data (no new data fetch)
python main.py --action summarize --workers 8

# Comprehensive multi-preset analysis
python main.py --action analyze --top-n 20
```

### Data Fetching Options
```bash
# S&P 500 universe
python main.py --action fetch --sp500 --period 5y

# Russell 2000 universe  
python main.py --action fetch --r2000 --period 3y

# ETF universe
python main.py --action fetch --etf --period 2y

# Custom symbols
python main.py --action fetch --symbols AAPL MSFT GOOGL --period 1y

# Incremental save mode (saves data per symbol vs batch)
python main.py --action fetch --sp500 --fetch-mode incremental --workers 4
```

## Screening Presets

9 predefined strategies in `config/screening_presets.json`:

- **deep_value**: Classic value with long-term potential (low P/E, P/B, positive 3Y returns)
- **high_growth**: Sustained growth with 3-year track record (high revenue/earnings growth, 15%+ 3Y returns)
- **fortress_balance_sheet**: Financial strength with 5-year performance (low debt, high liquidity, 20%+ 5Y returns)
- **short_term_momentum**: Pure momentum - recent price performance (1M/3M returns)
- **dividend_aristocrats**: High sustainable dividend yield with 5-year track record (4-15% yield, 10%+ 5Y returns)
- **low_volatility_stable**: Defensive strategy with consistent long-term performance (low volatility/beta, 5%+ 3Y, 15%+ 5Y returns)
- **small_cap_growth**: Small-cap growth with proven 3-year potential (300M-2B market cap, 20%+ 3Y returns)
- **long_term_compounders**: Exceptional 5-year wealth builders (50%+ 5Y returns, quality metrics)
- **turnaround_candidates**: Recent improvement after long-term underperformance (negative 3Y, positive 1Y returns)

## Custom Screening

Edit `config/screening_presets.json` to create custom strategies with filters and ranking criteria. Supports operators like `gt`, `lt`, `between`, `notnull` and fields like `pe_ratio`, `roe`, `market_cap`, `return_1y`, etc.

## Advanced Features

### Database Management
- **SQLite Backend**: Efficient storage and retrieval of price and fundamental data
- **Fundamentals Recalculation**: Use `--action summarize` to recalculate metrics from existing price data without re-fetching
- **Incremental Updates**: Support for incremental data fetching and storage

### Performance Optimization
- **Multi-threading**: Use `--workers N` to specify number of parallel threads for data processing
- **Batch vs Incremental**: Choose between batch mode (default) or incremental mode for data saving
- **Memory Efficient**: Database-first approach reduces memory usage for large datasets

### Data Sources
- **S&P 500**: Built-in support with `--sp500` flag
- **Russell 2000**: Small-cap universe with `--r2000` flag  
- **ETFs**: Exchange-traded funds universe with `--etf` flag
- **Custom Lists**: Specify any symbols with `--symbols` parameter

## Example Output

### Screening Results
```
=== SCREENING RESULTS ===
Rank Symbol  Score    P/E      ROE      Mkt Cap      Sector         
1    AAPL    87.3     28.7     36.9%    $3.0T       Technology     
2    MSFT    84.1     35.2     36.7%    $2.8T       Technology     
3    GOOGL   79.8     26.3     14.2%    $1.7T       Technology     
```

### Performance Report
```
=== PERFORMANCE REPORT: value_stocks_portfolio ===
Period: 2024-01-01 to 2024-07-11
Duration: 191 days

Total Return:      12.45%
Annualized Return: 24.89%
Volatility:        16.23%
Sharpe Ratio:      1.53
Max Drawdown:      -8.21%

=== BENCHMARK COMPARISON ===
Portfolio Return:  24.89%
Benchmark Return:  18.45%
Alpha:             6.44%
```