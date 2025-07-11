# Intelligent Stock Screener

A comprehensive stock screening system with real-time data, multiple strategies, and advanced overlap analysis.

## Features

- **Real Data**: Uses yfinance for current stock and fundamental data
- **8 Screening Strategies**: Value, growth, quality, momentum, long-term momentum, consistent performers, low-volatility growth, and quality dividend stocks
- **Multi-Preset Analysis**: Identifies stocks that score well across multiple strategies
- **Performance Tracking**: Backtesting with benchmark comparison and risk metrics
- **Extended Timeframes**: 1M, 3M, 6M, 1Y, 3Y return calculations with volatility metrics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline with S&P 500 data
python main.py --action full --sp500 --preset value_stocks --top-n 20

# Multi-preset overlap analysis
python analyze_presets.py
```

## Usage

### Basic Commands
```bash
# Fetch data only
python main.py --action fetch --sp500 --period 2y

# Screen stocks only (requires existing data)
python main.py --action screen --preset growth_stocks --top-n 15

# Backtest performance only
python main.py --action backtest --preset quality_stocks

# Multi-preset overlap analysis
python analyze_presets.py --min-presets 3
```

## Screening Presets

8 predefined strategies in `screener/screening_presets.json`:

- **value_stocks**: Low P/E and P/B ratios, market cap >$1B
- **growth_stocks**: High revenue/earnings growth, P/E <50
- **quality_stocks**: Strong ROE, low debt, good margins  
- **momentum_stocks**: Strong recent performance (1M/3M returns)
- **long_term_momentum**: Consistent multi-year performance
- **consistent_performers**: Multi-timeframe growth with volatility adjustment
- **low_volatility_growth**: Growth stocks with lower risk profiles
- **quality_dividend**: Sustainable dividend yields with strong fundamentals

## Custom Screening

Edit `screener/screening_presets.json` to create custom strategies with filters and ranking criteria. Supports operators like `gt`, `lt`, `between`, `notnull` and fields like `pe_ratio`, `roe`, `market_cap`, `return_1y`, etc.

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