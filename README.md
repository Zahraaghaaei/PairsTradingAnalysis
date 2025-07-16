
# PairsTradingAnalysis

This repository contains a comprehensive implementation of various pairs trading strategies, leveraging clustering techniques and cointegration analysis to identify and trade historically correlated assets. The project is structured to provide a clear pipeline from data preprocessing and pair identification to strategy execution and performance evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Strategies Implemented](#strategies-implemented)
  - [Finding Pairs](#1-finding-pairs-1-finding_pairsipynb)
  - [Threshold Pairs Trading Strategy](#2-threshold-pairs-trading-strategy-2-trading_on_pairsipynb)
  - [Mean Reverting Portfolio (MRP) Strategy](#3-mean-reverting-portfolio-mrp-strategy-3-mrpipynb)
  - [Weighted Pairs Trading Strategy](#4-weighted-pairs-trading-strategy-4-weighted_pairs_trading_strategyipynb)
- [Key Classes](#key-classes)
  - [DataProcessor.py](#1-DataProcessorpy)
  - [PairsTradingStrategy.py](#2-PairsTradingStrategypy)
  - [Trader.py](#3-Traderpy)
  - [MRP.py](#4-MRPpy)
- [Data](#data)
- [Requirements](#requirements)
- [Contributing](#Contributing)
- [Contact](#Contact)

## Project Overview

Pairs trading is a market-neutral strategy that capitalizes on the relative price movements of two historically correlated assets. This project automates the entire process, from identifying suitable pairs to simulating trading strategies and evaluating their performance. It explores both traditional threshold-based strategies and more advanced Mean Reverting Portfolio (MRP) with Weighted Pairs Trading approaches.

## Project Structure

```bash
PairsTradingAnalysis/
├── Data/
│   ├── new_pickle/
│   ├── Prices.xlsx
│   └── Tickers_and_Sectors.xlsx
├── classes/
│   ├── DataProcessor.py
│   ├── MRP.py
│   ├── PairsTradingStrategy.py
│   └── Trader.py
├── notebooks/
│   ├── 1. Finding_Pairs.ipynb
│   ├── 2. Trading_on_Pairs.ipynb
│   ├── 3. MRP.ipynb
│   └── 4. Weighted Pairs Trading Strategy.ipynb
├── README.md
└── requirements.txt
```

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-username/PairsTradingAnalysis.git
cd PairsTradingAnalysis
```
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt

```

## Usage

The project workflow is primarily driven by the Jupyter notebooks located in the `notebooks/` directory. These notebooks should be run in the specified order to ensure proper data flow and strategy execution

## Strategies Implemented

### 1. Finding Pairs (`1. Finding_Pairs.ipynb`)

This notebook focuses on the initial crucial step of identifying statistically significant pairs for trading.

#### What is Pairs Trading?

A market-neutral strategy that profits from the relative price movements between historically correlated assets.  
**Key Steps:**  
- Identify Cointegrated Pairs  
- Detect Divergence  
- Trade the Spread  
- Revert to Mean  

#### **Goal**

Implement a full pipeline for building a pairs trading strategy:
- Load and clean stock price data  
- Cluster stocks based on categories and sectors  
- Reduce noise with PCA  
- Cluster similar stocks using OPTICS  
- Identify statistically valid pairs using cointegration  
- Save reusable data with `pickle`  

#### **Methodology**

- **Data Processing:**  
  Loads historical price data, removes tickers with NaN values, and splits data into training (2007-01-01 to 2012-12-31) and testing (2013-01-01 to 2016-12-31) sets.

- **Sector and Category Clustering:**  
  Classifies tickers by Segment and Sector, converting labels to numerical values for quantitative analysis.

- **Pairs Identification:**  
  Uses a cointegration-based filtering method (Engle-Granger test with p-value < 0.01) to identify 297 statistically strong mean-reverting pairs.

- **Unsupervised Clustering (OPTICS):**  
  Applies PCA for dimensionality reduction (5 principal components; >12% variance explained by the first component), followed by OPTICS clustering. Identifies 24 natural stock groupings.

---

### 2. Threshold Pairs Trading Strategy (`2. Trading_on_Pairs.ipynb`)

This notebook applies a threshold-based strategy to the identified pairs.

#### What is Threshold Pairs Trading?

An enhancement to basic pairs trading that uses entry and exit signals based on spread deviations from the mean.

#### **Strategy Summary**

1. Find a stable pair  
2. Calculate the spread  
3. Compute the mean and standard deviation of the spread (from training data)  
4. Set thresholds (e.g., ±1 standard deviations)  
5. Generate trading signals  
   - Short if spread > upper threshold  
   - Long if spread < lower threshold  
6. Exit when the spread reverts to the mean  

#### **Workflow**

- Loads pre-identified pairs and ticker data  
- Applies threshold-based trading logic  
- Evaluates strategy performance on a test set  

#### **Key Features**

- Cost modeling: commissions, short fees, market impact  
- Position management: stabilization period, automatic sizing  
- Performance metrics: Sharpe ratio, max drawdown, total ROI  

---

### 3. Mean Reverting Portfolio (MRP) Strategy (`3. MRP.ipynb`)

This notebook implements an advanced statistical arbitrage strategy using optimized portfolios.

#### What is Mean Reverting Portfolio Trading?

A strategy that identifies groups of cointegrated assets and constructs a portfolio designed to profit from mean-reverting behavior.

#### **Strategy Core Mechanism**

- **Pair Selection:**  
  Uses both fundamental sector/category data and OPTICS clustering to identify equilibrium relationships.

- **Portfolio Optimization:**  
  Based on the **Majorization-Minimization (MM)** algorithm to minimize:
  - Mean-reversion strength: `U(w)`  
  - Return profile: `R(w)`  
  - Strategic sparsity: `S(w)`

- **Trading Execution:**  
  Opens positions when the spread diverges, closes when it converges, with ongoing portfolio monitoring.

#### **Workflow**

- Loads precomputed pairs  
- Performs optimization balancing `U(w)`, `R(w)`, and `S(w)`  
- Saves the resulting optimal portfolio weights  

---

### 4. Weighted Pairs Trading Strategy (`4. Weighted Pairs Trading Strategy.ipynb`)

This notebook explores a variation of the threshold-based strategy with dynamic weighting options.

#### **Goal**

Implement a threshold-based trading strategy with an emphasis on how applying weights to different pairs might impact performance.

#### **Workflow**

- Loads and preprocesses historical stock prices  
- Loads pre-identified pairs  
- Applies threshold-based logic (with potential weighting)  
- Evaluates strategy performance  
- Visualizes spread dynamics and signal behavior  

#### **Key Features**

- Similar performance evaluation features as `Trading_on_Pairs.ipynb`  
- Cost modeling, position sizing, and risk management  
- Flexible architecture for experimenting with different weighting schemes  

   
To open and run the notebooks:
```bash
jupyter notebook
```
This command will open a browser window with the Jupyter interface, from which you can navigate to the `notebooks/` directory and open each file.

## Key Classes

The `classes/` directory contains modular Python scripts that encapsulate core functionalities:

### 1. `DataProcessor.py`

- **Data Preparation:**  
  `split_data()`, `remove_tickers_with_nan()`

- **Pair Discovery:**  
  `get_candidate_pairs()`, `find_pairs()`

- **Validation Checks:**  
  `check_properties()`, `check_for_stationarity()`

- **Key Features:**  
  - Works with pre-clustered stocks  
  - Validates training/testing periods  
  - Tracks rejection reasons  
  - Optimized with subsampling  

---

### 2. `PairsTradingStrategy.py`  
*(Note: This file is assumed to include or relate to `PairsTradingPortfolio` as referenced in the weighted strategy notebook.)*

- **Core Functionality:**  
  - Implements fixed-beta pairs trading with configurable entry/exit thresholds  

- **Performance Tracking:**  
  - Calculates returns, Sharpe ratios, and account balances  

- **Risk Management:**  
  - Incorporates transaction costs, short fees, and position stabilization  

- **Key Features:**  
  - Cost modeling (commissions, short fees, market impact)  
  - Position management (stabilization, sizing, duration tracking)  
  - Comprehensive performance metrics (annualized/cumulative returns, win/loss, portfolio Sharpe, drawdown)  

---

### 3. `Trader.py`  
*(Note: Used in `2. Trading_on_Pairs.ipynb`, likely via `Trader.Trading()`.)*

- **Core Functionality:**  
  - Executes and evaluates trading strategies  
  - Includes: `apply_strategy()`, `threshold_strategy()`, `calculate_returns()`, `calculate_balance()`

- **Performance Analysis:**  
  - `analyze_results()`  
  - `_calculate_sharpe()`  
  - `_calculate_max_drawdown()`  
  - `_calculate_portfolio_sharpe()`

- **Supporting Utilities:**  
  - `trade_summary()`  
  - `_add_transaction_costs()`  
  - `_add_trading_duration()`  

---

### 4. `MRP.py`

- **Portfolio Optimization:**  
  Implements an advanced optimization framework for constructing mean-reverting portfolios using the **Majorization-Minimization (MM)** algorithm

- **Objective:**  
  - Maximize mean-reversion strength: `U(w)`  
  - Maximize return profile: `R(w)`  
  - Enforce strategic sparsity: `S(w)`

- **Key Features:**  
  - Computes optimal weights  
  - Tracks convergence history  
  - Provides methods for evaluating portfolio performance  


## Data

The `Data/` directory contains the following components essential for data preprocessing, clustering, and trading strategy execution:

- **`Prices.xlsx`**  
  Contains raw historical stock price data used for analysis and strategy simulation.

- **`Tickers_and_Sectors.xlsx`**  
  Metadata file that includes each stock's category and sector. This information is used for fundamental-based clustering in pair identification.

- **`new_pickle/`** *(Directory)*  
  Stores serialized intermediate data objects generated during notebook execution. These files are crucial for maintaining consistency and reusability across notebooks:
  - `df_prices`: Cleaned and preprocessed stock price DataFrame  
  - `pairs_category`, `pairs_OPTICS_unsupervised`: Lists of identified cointegrated stock pairs from different clustering methods  
  - `ticker_category_dict`, `ticker_segment_dict`: Dictionaries mapping tickers to their fundamental attributes  

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request.

---

## Contact

Vacancy – [https://www.linkedin.com/in/zahraaghaei95/](https://www.linkedin.com) – zahraaghaaei@gmail.com  
Project Link: [https://github.com/Zahraaghaaei/PairsTradingAnalysis](https://github.com/your-username/pairs-trading-strategy)
