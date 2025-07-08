# A Comparative Analysis of Mean-Variance Optimization vs. Hierarchical Risk Parity

This project provides a practical implementation and comparison of two prominent portfolio allocation strategies: the classic Mean-Variance Optimization (MVO) and the modern, machine-learning-based Hierarchical Risk Parity (HRP).

## Project Goal

The primary goal is to demonstrate a thorough understanding of portfolio management theories, their practical implementations, and their limitations. By comparing MVO and HRP through a robust backtesting framework, this project showcases critical thinking and a practical approach to portfolio construction.

## Core Concepts

*   **Mean-Variance Optimization (MVO):** A cornerstone of modern portfolio theory, MVO aims to maximize portfolio return for a given level of risk. It is implemented with the Ledoit-Wolf shrinkage estimator for a more robust covariance matrix. MVO is known for its sensitivity to input errors and can sometimes lead to concentrated portfolios.
*   **Hierarchical Risk Parity (HRP):** Developed by Marcos López de Prado, HRP is a novel approach that uses hierarchical clustering to address some shortcomings of MVO. It is less sensitive to input errors and tends to produce more diversified and stable portfolios by allocating weights based on asset relationships.

## Project Structure

The project is organized into several key components:

*   **`Portfolio` (Abstract Base Class):** Defines the common interface for portfolio strategies, including methods for getting weights and backtesting.
*   **`MVOPortfolio`:** Implements the Mean-Variance Optimization strategy, allowing for optimization based on Sharpe Ratio or minimum volatility.
*   **`HRPPortfolio`:** Implements the Hierarchical Risk Parity strategy, utilizing hierarchical clustering and recursive bisection for weight allocation.
*   **`Backtester`:** Manages the overall simulation process, including data acquisition, running backtests for different strategies, calculating performance metrics, and visualizing results.
*   **`config.json`:** Configuration file for backtesting parameters such as assets, date ranges, transaction costs, risk-free rate, and rebalancing frequency.

## Project Plan

This project will be executed in the following phases, ensuring that all tools and data sources are free of charge.

### 1. Environment Setup

*   **Programming Language:** Python 3.x
*   **Core Libraries:**
    *   `yfinance`: For downloading free historical price data from Yahoo Finance.
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations and linear algebra.
    *   `scipy`: For optimization (specifically, for MVO).
    *   `scikit-learn`: For hierarchical clustering (a key component of HRP) and Ledoit-Wolf covariance estimation.
    *   `matplotlib` / `seaborn`: For data visualization.

### 2. Data Acquisition

The `Backtester` class uses `yfinance` to download daily price data for a diverse set of assets specified in `config.json`. This allows for flexible testing with various asset classes.

### 3. Implementation Details

#### a. Mean-Variance Optimization (MVO)

The `MVOPortfolio` class calculates expected returns and uses the Ledoit-Wolf shrinkage estimator for the covariance matrix to enhance robustness. It then optimizes portfolio weights to either maximize the Sharpe Ratio or minimize volatility, subject to constraints (e.g., weights sum to 1, no short-selling).

#### b. Hierarchical Risk Parity (HRP)

The `HRPPortfolio` class constructs a hierarchical tree of assets based on their correlation using `AgglomerativeClustering`. It then applies quasi-diagonalization and recursive bisection to allocate portfolio weights, aiming for a more diversified and stable allocation compared to MVO.

### 4. Backtesting and Comparison

The `Backtester` class orchestrates the simulation:

1.  **Configuration Loading:** Parameters like `start_date`, `end_date`, `transaction_costs`, `risk_free_rate`, and `rebalancing_frequency` are loaded from `config.json`.
2.  **Data Retrieval:** Historical price data for specified assets is downloaded.
3.  **Strategy Execution:** Both MVO (Sharpe and Min Volatility) and HRP strategies are applied to the historical data.
4.  **Rebalancing:** Portfolios are rebalanced at the specified frequency (e.g., quarterly), and transaction costs are applied.
5.  **Performance Metrics:** The strategies are rigorously compared based on the following key metrics:
    *   **Sharpe Ratio:** Risk-adjusted return.
    *   **Maximum Drawdown:** Largest peak-to-trough decline.
    *   **Sortino Ratio:** Risk-adjusted return considering only downside volatility.
    *   **Calmar Ratio:** Annualized return divided by absolute maximum drawdown.
    *   **Herfindahl Index:** Measures portfolio concentration.
    *   **Turnover:** Indicates trading activity.

### 5. Analysis and Visualization

The results of the backtest are visualized using `matplotlib` to provide a clear comparison of the strategies. This includes:

*   **Portfolio Performance Comparison:** Cumulative returns over time (`images/performance_comparison.png`).
*   **Portfolio Weight Comparison:** Allocation weights for each strategy (`images/weights_comparison.png`).

## Skills Showcased

This project demonstrates proficiency in the following areas:

*   **Financial Theory:** Modern Portfolio Theory, Risk Management, Performance Measurement.
*   **Machine Learning:** Hierarchical Clustering, Covariance Estimation.
*   **Quantitative Analysis:** Linear Algebra, Optimization, Statistical Analysis.
*   **Programming:** Python (Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, yfinance).
*   **Software Engineering:** Object-Oriented Programming, Modular Design, Configuration Management.
*   **Critical Thinking:** Comparative Analysis, Evaluation of financial models, Understanding practical implications of theoretical concepts.
