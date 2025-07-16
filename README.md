# 📊 Pairs Trading Strategy using Clustering and Cointegration

This project develops a **market-neutral pairs trading strategy** by identifying statistically robust stock pairs using clustering (OPTICS) and cointegration. It includes modules for pair selection, backtesting, and strategy enhancement using Mean Reversion Portfolios (MRP).

---

## 🎯 Project Goals

- 🧹 Load and preprocess stock price data
- 📊 Cluster stocks by category and sector
- 📉 Reduce dimensionality using PCA
- 🤖 Cluster stocks using OPTICS
- 🔗 Identify cointegrated pairs
- 🧪 Apply Mean Reversion Portfolio (MRP) analysis
- 💹 Simulate trading strategies
- 💾 Save reusable data with `pickle`

---

## 🗃️ Folder Structure

```bash
pairs-trading-strategy/
├── Data/
│   ├── new_pickle/                      # Saved intermediate objects
│   ├── Prices.xlsx                      # Raw stock prices
│   └── Tickers_and_Sectors.xlsx         # Stock metadata
├── classes/
│   ├── DataProcessor.py                 # Data cleaning and preprocessing
│   ├── MRP.py                           # Mean Reversion Portfolio logic
│   ├── PairsTradingStrategy.py          # Core strategy implementation
│   └── Trader.py                        # Trade execution simulation
├── notebooks/
│   ├── 1. Finding_Pairs.ipynb         # Pair selection pipeline
│   ├── 2. Trading_on_Pairs.ipynb      # Trading logic on selected pairs
│   ├── 3. MRP.ipynb                   # MRP modeling and testing
│   └── 4. Weighted Pairs Trading Strategy.ipynb # Weighted version of the strategy
├── README.md
└── requirements.txt
```

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Zahraaghaaei/PairsTradingAnalysis.git
cd PairsTradingAnalysis
pip install -r requirements.txt
```

---

## 🧑‍💻 Usage

Start with the following notebooks in order:

1. `1. Finding_Pairs.ipynb` – Data processing, clustering, cointegration
2. `2. Trading_on_Pairs.ipynb` – Apply trading logic
3. `3. MRP.ipynb` – Build and evaluate MRPs
4. `4. Weighted Pairs Trading Strategy.ipynb` – Enhanced strategy with weight adjustments

---

## 🧪 Testing

Although this project is notebook-driven, you can modularize the code into the `classes/` folder and write unit tests using:

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request.

---

## 🌐 Contact

Vacancy – [https://www.linkedin.com/in/zahraaghaei95/](https://www.linkedin.com) – zahraaghaaei@gmail.com  
Project Link: [https://github.com/Zahraaghaaei/PairsTradingAnalysis](https://github.com/your-username/pairs-trading-strategy)
