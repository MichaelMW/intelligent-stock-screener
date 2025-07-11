# Standalone Stock Picker

A minimal, functional stock screener with real data and performance tracking.

## Features

- **Real Data**: Uses yfinance for reliable stock and fundamental data
- **Multi-Stock Processing**: Handles S&P 500 or custom stock universes
- **Extended Timeframes**: Supports 1m, 3m, 6m, 1y, 3y return calculations plus volatility metrics
- **Screening Criteria**: 8 predefined screens including long-term momentum, risk-adjusted strategies, and quality dividend stocks
- **Overlap Analysis**: Multi-preset analysis to identify high-quality candidates across strategies
- **Performance Tracking**: Retrospective analysis with benchmark comparison
- **Actionable Output**: Ranked stock lists with detailed metrics and comprehensive scoring

## Requirements

- Python 3.11 or higher (required for yfinance)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/MichaelMW/intelligent-stock-screener.git
cd intelligent-stock-screener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Command Line Interface**:
```bash
# Run full pipeline with default settings
python main.py --action full

# Screen S&P 500 for value stocks
python main.py --action full --sp500 --preset value_stocks --top-n 20
```

## Usage

### Command Line Interface

#### Fetch Data Only
```bash
python main.py --action fetch --symbols AAPL MSFT GOOGL
python main.py --action fetch --sp500 --period 2y
```

#### Screen Stocks Only
```bash
python main.py --action screen --preset growth_stocks --top-n 15
```

#### Backtest Performance Only
```bash
python main.py --action backtest --preset quality_stocks
```

### Multi-Preset Overlap Analysis

The system includes a powerful overlap analysis tool that runs all screening presets and analyzes which stocks appear across multiple strategies:

```bash
python analyze_presets.py
```

This generates:
- **overlap_analysis.csv**: Complete results with all preset scores and ranks for each stock
- Identification of high-quality candidates that score well across multiple screening strategies
- Comprehensive scoring data for informed investment decisions

The analysis output includes scores and ranks for all presets, allowing you to identify stocks that consistently perform well across different screening criteria.

## Screening Presets

The application comes with predefined screening presets located in `screener/screening_presets.json`. You can customize these or create new ones by modifying the JSON file.

### Available Presets

#### Core Strategies
- **value_stocks**: Low P/E and P/B ratios, good fundamentals
- **growth_stocks**: High revenue and earnings growth
- **quality_stocks**: Strong ROE, low debt, good margins
- **momentum_stocks**: Strong recent performance

#### Enhanced Long-Term Strategies
- **long_term_momentum**: Stocks with consistent long-term momentum (requires 1y+ and 3y+ returns with 6m+ gains)
- **consistent_performers**: Multi-timeframe growth across 6m, 1y, and 3y periods with volatility-adjusted scoring
- **low_volatility_growth**: Growth stocks with lower volatility profiles (growth + low volatility ranking)
- **quality_dividend**: Quality dividend stocks with sustainable yields (1.5-8%), inflation-beating growth (3%+), and strong fundamentals (ROE 12%+, manageable debt)

### Custom Screening Configuration

You can customize screening criteria by editing `screener/screening_presets.json`. The format is:

```json
{
  "your_preset_name": {
    "description": "Description of your screening strategy",
    "filters": [
      {
        "field": "pe_ratio",
        "operator": "lt",
        "value": 20
      },
      {
        "field": "market_cap",
        "operator": "gt", 
        "value": 1000000000
      }
    ],
    "ranking": [
      {
        "field": "pe_ratio",
        "weight": 2.0,
        "direction": "lower"
      },
      {
        "field": "roe",
        "weight": 1.0,
        "direction": "higher"
      }
    ],
    "top_n": 20
  }
}
```

#### Filter Operators
- `gt`: Greater than
- `lt`: Less than
- `gte`: Greater than or equal
- `lte`: Less than or equal
- `eq`: Equal
- `ne`: Not equal
- `between`: Between two values (use array: `[min, max]`)
- `notnull`: Field has a value (not null)

#### Ranking Directions
- `higher`: Higher values get better scores
- `lower`: Lower values get better scores

#### Available Fields
Common fields you can filter and rank by:
- `pe_ratio`, `pb_ratio`, `ps_ratio`, `peg_ratio`
- `market_cap`, `price`, `volume`
- `roe`, `roa`, `profit_margin`, `operating_margin`
- `debt_to_equity`, `current_ratio`, `quick_ratio`
- `revenue_growth`, `earnings_growth`, `dividend_yield`
- `return_1m`, `return_3m`, `return_6m`, `return_1y`, `return_3y`
- `volatility` (annualized volatility)
- `beta`, `sector`, `industry`

## Output Files

- `stock_data.json`: Raw fetched data
- `fundamentals.csv`: Fundamental metrics in CSV format
- `{preset}_results_{date}.csv`: Screening results
- `{preset}_performance_{date}.json`: Performance analysis

## Architecture

```
standalone_picker/
├── main.py                 # Main CLI application
├── analyze_presets.py      # Multi-preset overlap analysis
├── utils/
│   ├── data_fetcher.py     # yfinance data fetching (enhanced with extended timeframes)
│   └── performance_tracker.py # Portfolio performance analysis
├── screener/
│   ├── stock_screener.py   # Screening logic and filters
│   └── screening_presets.json # Screening configuration (8 presets)
├── data/                   # Output directory
└── requirements.txt        # Dependencies
```

## Key Components

### DataFetcher
- Fetches S&P 500 symbols from Wikipedia
- Uses yfinance for price and fundamental data
- Enhanced with extended timeframes (6m, 1y, 3y returns plus volatility)
- Handles batch processing with error handling

### StockScreener  
- Configurable filtering system
- Multi-criteria ranking
- Predefined screening strategies
- Export capabilities

### PerformanceTracker
- Portfolio performance calculation
- Benchmark comparison (S&P 500)
- Risk metrics (Sharpe ratio, max drawdown)
- Backtesting framework

## Example Output

```
=== TOP 10 STOCKS ===
Rank Symbol  Score    P/E      P/B      ROE      Mkt Cap      Sector         
1    BRK-B   94.2     12.3     1.2      18.5%    $678.9B      Financial      
2    JPM     89.7     11.8     1.4      15.2%    $456.2B      Financial      
3    WMT     85.1     25.1     2.8      22.1%    $567.3B      Consumer Def   
```

## Performance Report

```
=== PERFORMANCE REPORT: value_stocks_portfolio ===
Period: 2024-01-01 to 2024-07-10
Duration: 191 days

Total Return:      12.45%
Annualized Return: 24.89%
Volatility:        16.23%
Sharpe Ratio:      1.53
Max Drawdown:      -8.21%
Win Rate:          56.02%

=== BENCHMARK COMPARISON ===
Portfolio Return:  24.89%
Benchmark Return:  18.45%
Alpha:             6.44%
Outperformance:    Yes
```