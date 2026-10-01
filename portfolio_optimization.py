"""
Mean-Variance Portfolio Optimization on NASDAQ Stocks
======================================================
Research question:
    Can a mean-variance optimized portfolio of NASDAQ stocks outperform a
    passive benchmark (QQQ) on a risk-adjusted basis, out-of-sample?

Pipeline:
    1. Download price data (yfinance)
    2. Compute daily log returns, annualized mean/covariance
    3. Simulate random portfolios -> plot the efficient frontier
    4. Solve for Max-Sharpe and Min-Variance portfolios (constrained: no
       shorting, max 25% per name)
    5. Backtest out-of-sample with quarterly rebalancing
    6. Compare vs. equal-weight and vs. QQQ benchmark
    7. Save all plots + a metrics table to /output

Run:
    python portfolio_optimization.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "ADBE", "COST", "PEP"]
BENCHMARK = "QQQ"
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
TRAIN_END = "2022-06-30"     # optimize on data before this date
MAX_WEIGHT = 0.25            # no single stock > 25% of portfolio
RISK_FREE_RATE = 0.02        # annualized, for Sharpe ratio
REBALANCE_FREQ = "QE"        # quarterly rebalancing in the backtest (quarter-end)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 2. DATA
# ----------------------------------------------------------------------
def download_prices(tickers, start, end):
    """Download adjusted close prices. Raises if download fails/empty."""
    all_tickers = tickers + [BENCHMARK]
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.dropna(how="all").ffill().dropna()
    if data.empty:
        raise RuntimeError("No price data returned — check tickers/dates or network access.")
    return data


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------
# 3. PORTFOLIO MATH
# ----------------------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Annualized return and volatility for a given weight vector."""
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(weights.T @ (cov_matrix * 252) @ weights)
    return ret, vol


def neg_sharpe(weights, mean_returns, cov_matrix, rf):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - rf) / vol


def portfolio_vol(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(mean_returns, cov_matrix, objective, rf=RISK_FREE_RATE, max_weight=MAX_WEIGHT):
    """objective: 'sharpe' or 'min_vol'"""
    n = len(mean_returns)
    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init_guess = np.repeat(1 / n, n)

    if objective == "sharpe":
        result = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix, rf),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "min_vol":
        result = minimize(portfolio_vol, init_guess, args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        raise ValueError("objective must be 'sharpe' or 'min_vol'")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def random_portfolios(mean_returns, cov_matrix, n_portfolios=8000, max_weight=MAX_WEIGHT):
    """Monte Carlo simulation of random weight combos for the frontier plot."""
    n = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        w = np.clip(w, 0, max_weight)
        w = w / w.sum()
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = (ret - RISK_FREE_RATE) / vol
    return results


# ----------------------------------------------------------------------
# 4. PERFORMANCE METRICS
# ----------------------------------------------------------------------
def compute_metrics(cum_returns_series, rf=RISK_FREE_RATE):
    daily_ret = cum_returns_series.pct_change().dropna()
    total_return = cum_returns_series.iloc[-1] / cum_returns_series.iloc[0] - 1
    n_years = len(daily_ret) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    running_max = cum_returns_series.cummax()
    drawdown = cum_returns_series / running_max - 1
    max_dd = drawdown.min()
    return {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


# ----------------------------------------------------------------------
# 5. BACKTEST WITH QUARTERLY REBALANCING
# ----------------------------------------------------------------------
def backtest_strategy(prices, tickers, weight_fn, train_window_years=3, freq=REBALANCE_FREQ):
    """
    Walk-forward backtest: at each rebalance date, look back `train_window_years`
    of data, compute weights via weight_fn(mean_returns, cov_matrix), then hold
    those weights until the next rebalance date. Returns a cumulative value series.
    """
    rets = log_returns(prices[tickers])
    rebalance_dates = rets.resample(freq).first().index
    rebalance_dates = rebalance_dates[rebalance_dates > rets.index[0] + pd.DateOffset(years=train_window_years)]

    portfolio_value = pd.Series(index=rets.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    current_weights = np.repeat(1 / len(tickers), len(tickers))

    for i in range(1, len(rets)):
        date = rets.index[i]
        if date in rebalance_dates:
            lookback_start = date - pd.DateOffset(years=train_window_years)
            train = rets.loc[lookback_start:date]
            if len(train) > 60:  # enough history to estimate covariance
                mean_r = train.mean()
                cov_m = train.cov()
                try:
                    current_weights = weight_fn(mean_r.values, cov_m.values)
                except Exception:
                    pass  # keep previous weights if optimization fails
        day_ret = np.dot(current_weights, rets.iloc[i].values)
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * np.exp(day_ret)

    portfolio_value.iloc[0] = 1.0
    return portfolio_value.ffill()


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
def main():
    print("Downloading price data...")
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    print(f"Data shape: {prices.shape}, date range {prices.index.min().date()} to {prices.index.max().date()}")

    rets = log_returns(prices[TICKERS])
    train_rets = rets.loc[:TRAIN_END]
    mean_returns = train_rets.mean()
    cov_matrix = train_rets.cov()

    # --- Efficient frontier (Monte Carlo) ---
    print("Simulating random portfolios for the efficient frontier...")
    sim = random_portfolios(mean_returns.values, cov_matrix.values)

    max_sharpe_w = optimize_portfolio(mean_returns.values, cov_matrix.values, "sharpe")
    min_vol_w = optimize_portfolio(mean_returns.values, cov_matrix.values, "min_vol")
    ms_ret, ms_vol = portfolio_performance(max_sharpe_w, mean_returns.values, cov_matrix.values)
    mv_ret, mv_vol = portfolio_performance(min_vol_w, mean_returns.values, cov_matrix.values)

    plt.figure(figsize=(9, 6))
    plt.scatter(sim[0], sim[1], c=sim[2], cmap="viridis", s=6, alpha=0.5)
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(ms_vol, ms_ret, c="red", marker="*", s=300, label="Max Sharpe")
    plt.scatter(mv_vol, mv_ret, c="blue", marker="*", s=300, label="Min Volatility")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier — NASDAQ Portfolio (in-sample, training data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "efficient_frontier.png"), dpi=150)
    plt.close()
    print("Saved efficient_frontier.png")

    print("\nMax Sharpe weights:")
    print(pd.Series(max_sharpe_w, index=TICKERS).round(3).sort_values(ascending=False))
    print("\nMin Volatility weights:")
    print(pd.Series(min_vol_w, index=TICKERS).round(3).sort_values(ascending=False))

    # --- Out-of-sample walk-forward backtest ---
    print("\nRunning walk-forward backtest (this compares strategies fairly, "
          "re-optimizing each quarter using only past data)...")

    max_sharpe_fn = lambda m, c: optimize_portfolio(m, c, "sharpe")
    min_vol_fn = lambda m, c: optimize_portfolio(m, c, "min_vol")
    equal_weight_fn = lambda m, c: np.repeat(1 / len(TICKERS), len(TICKERS))

    strat_max_sharpe = backtest_strategy(prices, TICKERS, max_sharpe_fn)
    strat_min_vol = backtest_strategy(prices, TICKERS, min_vol_fn)
    strat_equal = backtest_strategy(prices, TICKERS, equal_weight_fn)

    bench_rets = log_returns(prices[[BENCHMARK]])[BENCHMARK]
    bench_value = np.exp(bench_rets.cumsum())
    bench_value = bench_value.reindex(strat_max_sharpe.index).ffill()
    bench_value = bench_value / bench_value.iloc[0]

    # --- Plot cumulative performance ---
    plt.figure(figsize=(10, 6))
    plt.plot(strat_max_sharpe, label="Max Sharpe (rebalanced quarterly)")
    plt.plot(strat_min_vol, label="Min Volatility (rebalanced quarterly)")
    plt.plot(strat_equal, label="Equal Weight (rebalanced quarterly)")
    plt.plot(bench_value, label=f"{BENCHMARK} (benchmark)", linestyle="--", color="black")
    plt.ylabel("Growth of $1")
    plt.title("Walk-Forward Backtest: Optimized Portfolios vs. Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "backtest_performance.png"), dpi=150)
    plt.close()
    print("Saved backtest_performance.png")

    # --- Drawdown plot ---
    plt.figure(figsize=(10, 4))
    for series, label in [(strat_max_sharpe, "Max Sharpe"), (bench_value, BENCHMARK)]:
        dd = series / series.cummax() - 1
        plt.plot(dd, label=label)
    plt.ylabel("Drawdown")
    plt.title("Drawdown: Max Sharpe Strategy vs. Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "drawdown.png"), dpi=150)
    plt.close()
    print("Saved drawdown.png")

    # --- Metrics table ---
    metrics = pd.DataFrame({
        "Max Sharpe": compute_metrics(strat_max_sharpe),
        "Min Volatility": compute_metrics(strat_min_vol),
        "Equal Weight": compute_metrics(strat_equal),
        BENCHMARK: compute_metrics(bench_value),
    }).T
    metrics.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))
    print("\nPerformance summary (out-of-sample, walk-forward):")
    print(metrics.round(4))
    print("\nSaved performance_metrics.csv")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
