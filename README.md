# Trader Performance vs Market Sentiment

## Objective

This project analyzes how market sentiment (Fear and Greed) influences trader performance and behavior. The goal is to identify patterns that can inform better trading decisions.

---

## Dataset

Two datasets are used in this analysis:

1. Bitcoin Fear and Greed Index

   * Contains daily sentiment classification (Fear, Neutral, Greed)

2. Historical Trader Data (Hyperliquid)

   * Includes trade-level data such as account, position size, direction, and closed PnL

Due to file size limitations, datasets are not included in this repository. They can be accessed from the links provided in the assignment.

---

## Dataset Access

Due to file size limitations, datasets are not included in this repository.

They can be downloaded from:

* Fear & Greed Index: https://drive.google.com/file/d/1yR62o2XoTjaO-ChSE1aWRKL4fQWlq2T_/view?usp=sharing
* Historical Trader Data: https://drive.google.com/file/d/1-XSoM1P-f45ZYXS319tIu72WsLl0W2O_/view?usp=sharing

Place the files in the project directory before running the code.

---

## Methodology

### Data Preparation

* Loaded both datasets using pandas
* Converted timestamps to datetime format
* Aligned datasets on a daily level using the date field
* Handled missing values and duplicates

### Feature Engineering

* Daily PnL per trader
* Win rate (ratio of profitable trades)
* Average trade size
* Leverage proxy
* Trade frequency (number of trades per day)
* Long/short ratio

### Segmentation

* High vs Low leverage traders
* Frequent vs Infrequent traders
* Net Winners vs Net Losers

---

## Analysis

### Performance by Sentiment

* Compared average daily PnL and win rate across Fear, Neutral, and Greed conditions
* Identified differences in trading outcomes under varying market sentiment

### Behavioral Patterns

* Analyzed changes in trade frequency, leverage usage, and position sizing
* Observed shifts in trader behavior depending on sentiment conditions

### Segment-Based Analysis

* Evaluated how different trader segments perform under different sentiment regimes
* Compared performance consistency across segments

---

## Key Insights

1. Trader performance varies across market sentiment conditions, with noticeable differences in PnL and win rate between Fear and Greed periods.

2. Trader behavior changes with sentiment. Metrics such as trade frequency, leverage, and position size show variation depending on market conditions.

3. High leverage traders tend to exhibit more aggressive behavior, while consistent performers maintain relatively stable performance across different sentiment phases.

---

## Strategy Recommendations

1. During Fear conditions:

   * Reduce leverage and control risk exposure
   * Avoid overtrading and focus on selective entries

2. During Greed conditions:

   * Increase position size cautiously
   * Higher trade activity can be beneficial for experienced traders

---

## Outputs

The project generates multiple visualizations, including:

* PnL and win rate comparisons by sentiment
* Behavioral metrics analysis
* Distribution of trader performance
* Segment-based performance heatmaps
* Cumulative PnL trends over time

---

## How to Run

Install dependencies:
pip install pandas numpy matplotlib seaborn

Run the script:
python analysis.py

---

## Conclusion

Market sentiment has a measurable impact on trader performance and behavior. Understanding these patterns can help in designing more adaptive and risk-aware trading strategies.
