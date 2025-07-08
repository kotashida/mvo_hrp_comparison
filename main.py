import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

class Portfolio:
    """An abstract base class for a portfolio allocation strategy."""
    def __init__(self, name, assets):
        self.name = name
        self.assets = assets
        self.weights = None

    def get_weights(self, returns, risk_free_rate=0, objective='sharpe'):
        raise NotImplementedError

    def backtest(self, returns, transaction_costs, rebalancing_frequency, risk_free_rate=0, objective=None):
        # Rebalance the portfolio at the specified frequency
        rebalancing_dates = returns.resample(rebalancing_frequency).first().index
        portfolio_returns = pd.Series(index=returns.index)
        last_weights = pd.Series(0, index=self.assets)
        total_turnover = 0

        for i in range(len(rebalancing_dates) - 1):
            start_date = rebalancing_dates[i]
            end_date = rebalancing_dates[i+1]
            
            # Get the returns for the current period
            period_returns = returns.loc[start_date:end_date]
            
            # Calculate the new weights
            self.get_weights(period_returns, risk_free_rate, objective)
            
            # Calculate the turnover
            turnover = (self.weights - last_weights).abs().sum() / 2
            total_turnover += turnover
            
            # Calculate the transaction costs
            costs = turnover * transaction_costs
            
            # Calculate the portfolio returns for the period
            period_portfolio_returns = (period_returns * self.weights).sum(axis=1) - costs
            portfolio_returns.loc[start_date:end_date] = period_portfolio_returns
            
            # Update the last weights
            last_weights = self.weights
            
        return portfolio_returns.dropna(), total_turnover

class MVOPortfolio(Portfolio):
    """
    A portfolio that uses Mean-Variance Optimization.
    """
    def get_weights(self, returns, risk_free_rate=0, objective='sharpe'):
        expected_returns = returns.mean() * 252
        
        # Ledoit-Wolf shrinkage estimator for covariance matrix
        lw = LedoitWolf()
        cov_matrix = pd.DataFrame(lw.fit(returns).covariance_ * 252, index=returns.columns, columns=returns.columns)

        if objective == 'sharpe':
            def objective_function(weights, expected_returns, cov_matrix, risk_free_rate):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_volatility == 0:
                    return 0
                return -(portfolio_return - risk_free_rate) / portfolio_volatility
        elif objective == 'min_volatility':
            def objective_function(weights, expected_returns, cov_matrix, risk_free_rate):
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_volatility
        else:
            raise ValueError("Invalid objective. Choose 'sharpe' or 'min_volatility'.")

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.assets)))
        initial_weights = np.array([1/len(self.assets)] * len(self.assets))

        result = minimize(objective_function, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = pd.Series(result.x, index=self.assets)

class HRPPortfolio(Portfolio):
    """A portfolio that uses Hierarchical Risk Parity."""
    def get_weights(self, returns, risk_free_rate=0, objective=None):
        cov_matrix = returns.cov() * 252
        corr_matrix = returns.corr()

        dist = np.sqrt((1 - corr_matrix) / 2)
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='single').fit(dist)
        link = self.get_linkage_matrix(model)

        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr_matrix.index[sort_ix].tolist()

        hrp_weights = self.get_rec_bipart(cov_matrix, sort_ix)
        self.weights = hrp_weights.sort_index()

    def get_linkage_matrix(self, model):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, np.zeros(model.children_.shape[0]), counts]).astype(float)
        return linkage_matrix

    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def get_rec_bipart(self, cov_matrix, sort_ix):
        weights = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = self.get_cluster_var(cov_matrix, c_items0)
                c_var1 = self.get_cluster_var(cov_matrix, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                weights[c_items0] *= alpha
                weights[c_items1] *= 1 - alpha
        return weights

    def get_cluster_var(self, cov_matrix, cluster_items):
        cov_slice = cov_matrix.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)
        weights /= weights.sum()
        cluster_var = np.dot(weights.T, np.dot(cov_slice, weights))
        return cluster_var

class Backtester:
    """A class for backtesting and comparing portfolio strategies."""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.assets = self.config['assets']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.transaction_costs = self.config['transaction_costs']
        self.risk_free_rate = self.config['risk_free_rate']
        self.rebalancing_frequency = self.config['rebalancing_frequency']
        # Map descriptive frequency to pandas frequency string
        self.pandas_rebalancing_frequency = {
            "quarterly": "QE",
            "monthly": "ME",
            "yearly": "YE"
        }.get(self.rebalancing_frequency.lower(), "QE") # Default to quarterly if not found
        self.returns = self.get_returns()

    def get_returns(self):
        data = yf.download(self.assets, start=self.start_date, end=self.end_date)
        close_prices = data['Close']
        return close_prices.pct_change().dropna()

    def run(self):
        # Create the portfolio strategies
        mvo_sharpe_portfolio = MVOPortfolio('MVO (Sharpe)', self.assets)
        mvo_min_vol_portfolio = MVOPortfolio('MVO (Min Vol)', self.assets)
        hrp_portfolio = HRPPortfolio('HRP', self.assets)

        # Run the backtests
        mvo_sharpe_returns, mvo_sharpe_turnover = mvo_sharpe_portfolio.backtest(self.returns, self.transaction_costs, self.pandas_rebalancing_frequency, self.risk_free_rate, objective='sharpe')
        mvo_min_vol_returns, mvo_min_vol_turnover = mvo_min_vol_portfolio.backtest(self.returns, self.transaction_costs, self.pandas_rebalancing_frequency, self.risk_free_rate, objective='min_volatility')
        hrp_returns, hrp_turnover = hrp_portfolio.backtest(self.returns, self.transaction_costs, self.pandas_rebalancing_frequency, self.risk_free_rate)

        # Calculate performance metrics
        mvo_sharpe_cum_returns, mvo_sharpe_sharpe, mvo_sharpe_max_dd, mvo_sharpe_sortino, mvo_sharpe_calmar, mvo_sharpe_herfindahl = self.calculate_performance_metrics(mvo_sharpe_returns, mvo_sharpe_portfolio.weights, self.risk_free_rate)
        mvo_min_vol_cum_returns, mvo_min_vol_sharpe, mvo_min_vol_max_dd, mvo_min_vol_sortino, mvo_min_vol_calmar, mvo_min_vol_herfindahl = self.calculate_performance_metrics(mvo_min_vol_returns, mvo_min_vol_portfolio.weights, self.risk_free_rate)
        hrp_cum_returns, hrp_sharpe, hrp_max_dd, hrp_sortino, hrp_calmar, hrp_herfindahl = self.calculate_performance_metrics(hrp_returns, hrp_portfolio.weights, self.risk_free_rate)

        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"MVO (Sharpe) Sharpe Ratio: {mvo_sharpe_sharpe:.4f}")
        print(f"MVO (Sharpe) Max Drawdown: {mvo_sharpe_max_dd:.4f}")
        print(f"MVO (Sharpe) Sortino Ratio: {mvo_sharpe_sortino:.4f}")
        print(f"MVO (Sharpe) Calmar Ratio: {mvo_sharpe_calmar:.4f}")
        print(f"MVO (Sharpe) Herfindahl Index: {mvo_sharpe_herfindahl:.4f}")
        print(f"MVO (Sharpe) Turnover: {mvo_sharpe_turnover:.4f}")

        print(f"\nMVO (Min Vol) Sharpe Ratio: {mvo_min_vol_sharpe:.4f}")
        print(f"MVO (Min Vol) Max Drawdown: {mvo_min_vol_max_dd:.4f}")
        print(f"MVO (Min Vol) Sortino Ratio: {mvo_min_vol_sortino:.4f}")
        print(f"MVO (Min Vol) Calmar Ratio: {mvo_min_vol_calmar:.4f}")
        print(f"MVO (Min Vol) Herfindahl Index: {mvo_min_vol_herfindahl:.4f}")
        print(f"MVO (Min Vol) Turnover: {mvo_min_vol_turnover:.4f}")

        print(f"\nHRP Sharpe Ratio: {hrp_sharpe:.4f}")
        print(f"HRP Max Drawdown: {hrp_max_dd:.4f}")
        print(f"HRP Sortino Ratio: {hrp_sortino:.4f}")
        print(f"HRP Calmar Ratio: {hrp_calmar:.4f}")
        print(f"HRP Herfindahl Index: {hrp_herfindahl:.4f}")
        print(f"HRP Turnover: {hrp_turnover:.4f}")

        # Plot results
        self.plot_performance(mvo_sharpe_cum_returns, mvo_min_vol_cum_returns, hrp_cum_returns)
        self.plot_weights(mvo_sharpe_portfolio.weights, mvo_min_vol_portfolio.weights, hrp_portfolio.weights)

        print("\nPlots saved to images/performance_comparison.png and images/weights_comparison.png")

    def calculate_performance_metrics(self, returns, weights, risk_free_rate):
        cumulative_returns = (1 + returns).cumprod() - 1
        
        excess_returns = returns - (risk_free_rate / 252) # Daily risk-free rate
        sharpe_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        max_drawdown = drawdown.min()
        
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (excess_returns.mean() * 252) / downside_std
        
        annualized_return = returns.mean() * 252
        calmar_ratio = annualized_return / abs(max_drawdown)

        # Herfindahl Index
        herfindahl_index = (weights**2).sum()
        
        return cumulative_returns, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, herfindahl_index

    def plot_performance(self, mvo_sharpe_returns, mvo_min_vol_returns, hrp_returns):
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.figure(figsize=(12, 6))
        plt.plot(mvo_sharpe_returns, label='MVO (Sharpe)')
        plt.plot(mvo_min_vol_returns, label='MVO (Min Vol)')
        plt.plot(hrp_returns, label='HRP')
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig('images/performance_comparison.png')
        plt.close()

    def plot_weights(self, mvo_sharpe_weights, mvo_min_vol_weights, hrp_weights):
        if not os.path.exists('images'):
            os.makedirs('images')
        weights_df = pd.DataFrame({'MVO (Sharpe)': mvo_sharpe_weights, 'MVO (Min Vol)': mvo_min_vol_weights, 'HRP': hrp_weights})
        weights_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Portfolio Weight Comparison')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.savefig('images/weights_comparison.png')
        plt.close()

if __name__ == "__main__":
    backtester = Backtester('config.json')
    backtester.run()