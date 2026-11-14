#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author   : Vitor Reis
Document : Exam Solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

folder_path = r"C:\Users\vitor\Documents\LBS Classes\LBS Python Classes\Classes\e517 - Python for Finance\Exams\Take Home Exam"
temp_CMD_BRENT_EU = pd.read_csv(rf"{folder_path}\CMD_BRENT_EU.csv", dtype={0: str}, index_col=0)
temp_CMD_GLOB = pd.read_csv(rf"{folder_path}\CMD_GLOB.csv", dtype={0: str}, index_col=0)
temp_CMD_GOLD_GLOB = pd.read_csv(rf"{folder_path}\CMD_GOLD_GLOB.csv", dtype={0: str}, index_col=0)
temp_EQY_EM = pd.read_csv(rf"{folder_path}\EQY_EM.csv", dtype={0: str}, index_col=0)
temp_EQY_SPY_US = pd.read_csv(rf"{folder_path}\EQY_SPY_US.csv", dtype={0: str}, index_col=0)
temp_FX_USD = pd.read_csv(rf"{folder_path}\FX_USD.csv", dtype={0: str}, index_col=0)
temp_TLT_US = pd.read_csv(rf"{folder_path}\TLT_US.csv", dtype={0: str}, index_col=0)

#%%
###############################################################################
######                             QUESTION 1                            ######
###############################################################################

########################
###### QUESTION A ######
########################

temp_CMD_BRENT_EU.columns = ["CMD_BRENT_EU"]
temp_CMD_GLOB.columns = ["CMD_GLOB"]
temp_CMD_GOLD_GLOB.columns = ["CMD_GOLD_GLOB"]
temp_EQY_EM.columns = ["EQY_EM"]
temp_EQY_SPY_US.columns = ["EQY_SPY_US"]
temp_FX_USD.columns = ["FX_USD"]
temp_TLT_US.columns = ["TLT_US"]

# Merge all data, convert prices to numeric, remove NA 
prices = pd.concat(
    [temp_CMD_BRENT_EU, temp_CMD_GLOB, temp_CMD_GOLD_GLOB,
     temp_EQY_EM, temp_EQY_SPY_US, temp_FX_USD, temp_TLT_US], axis=1
).apply(pd.to_numeric, errors='coerce').dropna()
prices.index = pd.to_datetime(prices.index, dayfirst=True)

# Sort the Prices by Date
prices = prices.sort_index()
print("Prices:\n\n", prices.head())

# Calculate Returns
returns = prices.pct_change().dropna()
print("\n\nReturns:\n\n", returns.head())

# Rebased prices plot
rebased = 100 * prices / prices.iloc[0]
rebased.plot(title="Rebased Prices (Base = 100)", figsize=(12, 6))
plt.xlabel("Date")
plt.ylabel("Rebased Price")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Drawdown plot
cummax = prices.cummax()                # running maximum
drawdown = (prices - cummax) / cummax   # drawdown            
drawdown.plot(title="Drawdowns", figsize=(12, 6))
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Compound annual growth rate (CAGR) % - assume number of years = 20
n_years = 20
cagr = np.power(prices.iloc[-1] / prices.iloc[0], 1 / n_years) - 1
cagr.name = "CAGR"

# Annualised Volatility %
vol_annualised = returns.std() * np.sqrt(252)
vol_annualised.name = "Annualised Volatility"

# Annualised Sharpe Ratio (Rf = 0)
daily_mean = returns.mean()
daily_std = returns.std()
sharpe_ratio_annualised = (daily_mean / daily_std) * np.sqrt(252)
sharpe_ratio_annualised.name = "Sharpe Rt (Yr)"

# Maximum drawdown %
max_drawdown = drawdown.min()
max_drawdown.name = "Maximum Drawdown"

# Skewness
skewness = returns.skew()
skewness.name = "Skewness"

# Kurtosis
kurtosis = returns.kurtosis()
kurtosis.name = "Kurtosis"

# Stats Summary
stats_summary = pd.concat([cagr, vol_annualised, sharpe_ratio_annualised, 
                           max_drawdown, skewness, kurtosis], axis=1)
print("\n\nStats Summary:\n\n", stats_summary)

########################
###### QUESTION B ######
########################

# Rolling 12-month return (approx 252 trading days)
returns_12m = (1 + returns).rolling(252).apply(np.prod, raw=True) - 1
returns_12m = returns_12m.dropna()

# Missing value check
print("\nMissing values check:\n\n", returns_12m.isna().sum())

# Signal: +1 if positive, -1 if negative
signal = np.where(returns_12m > 0, 1, -1)
signal = pd.DataFrame(signal, index=returns_12m.index, columns=returns_12m.columns)

# Rolling 60-day annualised volatility
rolling_vol = returns.rolling(60).std() * np.sqrt(252)

# Inverse-volatility weighted signed positions (target vol = 10%)
target_vol = 0.10
weights = (target_vol / rolling_vol) * signal
weights = weights.dropna()

print("\nReturns 12m:\n", returns_12m.head())
print("\nSignal:\n", signal.head())
print("\nRolling Vol (first valid rows):\n", rolling_vol.dropna().head())
print("\nWeights:\n", weights.head())

########################
###### QUESTION C ######
########################

# Apply weights to next period's returns - shift weights by 1 to avoid look-ahead bias
strategy_asset_returns = weights.shift(1) * returns

# Sum across assets; min_count=1 ensures rows with all NaN become NaN, not 0
strategy_returns = strategy_asset_returns.sum(axis=1, min_count=1).dropna()

# Cumulative wealth index
cumulative_return = (1 + strategy_returns).cumprod()

# Total cumulative return %
total_cumulative_return = cumulative_return.iloc[-1] - 1

# CAGR % - assuming 20 years as said in the question
n_years_strat = 20
cagr_strat = cumulative_return.iloc[-1] ** (1 / n_years_strat) - 1

# Annualised Volatility %
vol_strat = strategy_returns.std() * np.sqrt(252)

# Sharpe Ratio (Rf = 0)
sharpe_strat = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

# Max Drawdown %
cummax_strat = cumulative_return.cummax()
drawdown_strat = (cumulative_return - cummax_strat) / cummax_strat
max_dd_strat = drawdown_strat.min()

# Strategy stats summary
strategy_stats = pd.Series({
    "Cumulative Return": total_cumulative_return,
    "CAGR": cagr_strat,
    "Annualised Volatility": vol_strat,
    "Sharpe Ratio": sharpe_strat,
    "Max Drawdown": max_dd_strat})

print("\nStrategy Returns:\n", strategy_returns.head())
print("\nStrategy Cumulative Return:\n", cumulative_return.head())
print("\nStrategy Stats:\n", strategy_stats)

########################
###### QUESTION D ######
########################

issues = """
3 Implementation Issues Not Addressed:

1. Transaction costs: The strategy assumes frictionless trading. In reality, 
   every time the signal flips (+1 to -1 or vice versa), you incur bid-ask 
   spreads, commissions, and market impact costs that erode returns.

2. No slippage / execution delay: we assume trades execute at exact closing 
   prices, but in real world there's slippage between the signal price and 
   the actual fill price.

3. No position sizing constraint / leverage limit: The inverse-volatility 
   weights can sum to well above 100%, meaning the strategy implicitly uses 
   leverage. In practice, there would be margin requirements and borrowing 
   costs that are not accounted for.
"""
print(issues)


#%%
###############################################################################
######                             QUESTION 2                            ######
###############################################################################

returns_q2 = pd.read_csv(rf"{folder_path}\RETURNS.CSV", dtype={0: str}, index_col=0)
returns_q2 = returns_q2.apply(pd.to_numeric, errors='coerce')

# Mean returns and covariance matrix
mu = returns_q2.mean()
vcov = returns_q2.cov()
n_assets = len(mu)

print("Mean returns:\n", mu)
print("\nCovariance matrix:\n", vcov)

########################
###### QUESTION 2A #####
########################

# Objective: minimise portfolio variance
def port_vol(w):
    return np.dot(w, np.dot(vcov, w))

# Constraints: weights sum to 1, target return = 10%
target_return = 0.10
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_return}
]

# Initial guess: equal weights
w0 = np.ones(n_assets) / n_assets

# Minimise (no bounds = short selling allowed)
res_2a = minimize(port_vol, w0, method='SLSQP', constraints=constraints)
print(f"\nOptimisation success: {res_2a.success} | {res_2a.message}")

print(f"\n2a) Optimal weights for 10% target return:")
for i, col in enumerate(returns_q2.columns):
    print(f"  {col}: {res_2a.x[i]:.4f}")

#########################
###### QUESTION 2B ######
#########################

vol_2a = np.sqrt(np.dot(res_2a.x, np.dot(vcov, res_2a.x)))
print(f"\n2b) Expected volatility: {vol_2a:.4%}")

#########################
###### QUESTION 2C ######
#########################

# Portfolio mean and vol from Q2a
port_mean = np.dot(res_2a.x, mu)
port_vol_2a = np.sqrt(np.dot(res_2a.x, np.dot(vcov, res_2a.x)))

# Monte Carlo: simulate annual returns assuming normal distribution
np.random.seed(42)
n_sims = 1_000_000
simulated_returns = np.random.normal(port_mean, port_vol_2a, n_sims)

# Probability of return below -15%
prob_below = np.mean(simulated_returns < -0.15)
print(f"\n2c) Probability of return below -15%: {prob_below:.4%}")

#########################
###### QUESTION 2D ######
#########################

# Objective: maximise return for target volatility of 35%
def neg_return(w):
    return -np.dot(w, mu)

target_vol_2d = 0.35
constraints_2d = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w, np.dot(vcov, w))) - target_vol_2d}
]

res_2d = minimize(neg_return, w0, method='SLSQP', constraints=constraints_2d)
print(f"\nOptimisation success: {res_2d.success} | {res_2d.message}")

print(f"\n2d) Optimal weights for 35% target volatility:")
for i, col in enumerate(returns_q2.columns):
    print(f"  {col}: {res_2d.x[i]:.4f}")
print(f"  Expected return: {np.dot(res_2d.x, mu):.4%}")

#########################
###### QUESTION 2E ######
#########################

# Minimise variance with target return = 9%, long-only (no short positions)
target_return_2e = 0.09
constraints_2e = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_return_2e}
]

# Bounds: each weight between 0 and 1 (no short selling)
bounds_2e = [(0, 1)] * n_assets

res_2e = minimize(port_vol, w0, method='SLSQP', constraints=constraints_2e, bounds=bounds_2e)
print(f"\nOptimisation success: {res_2e.success} | {res_2e.message}")

print(f"\n2e) Optimal weights for 9% target return (long-only):")
for i, col in enumerate(returns_q2.columns):
    print(f"  {col}: {res_2e.x[i]:.4f}")
print(f"  Expected volatility: {np.sqrt(np.dot(res_2e.x, np.dot(vcov, res_2e.x))):.4%}")


#%%
###############################################################################
######                             QUESTION 3                            ######
###############################################################################

q3_answer = """
PROPOSAL A: Remove assets with the lowest expected return

This is not a good rule by itself. An asset with a low expected return can still 
be useful if it helps diversify the portfolio. For example, bonds may have lower 
expected returns than equities, but they can reduce portfolio risk because they 
often behave differently from equities. 

The problem with this proposal is that it only looks at return. It ignores how 
the asset moves together with the other assets. Removing low-return assets could 
make the portfolio less diversified and more risky.


PROPOSAL B: Remove assets with the highest volatility

This is also not a good rule by itself. A high-volatility asset is not always bad. 
If it does not move in the same direction as the rest of the portfolio, it can 
still improve diversification.

The problem with this proposal is that it only looks at risk in isolation. It 
does not consider correlation with the other assets. In a portfolio, what matters 
is not only how risky one asset is, but how much risk it adds to the whole 
portfolio.


PROPOSAL C: Remove assets with the lowest Sharpe Ratio

This is the best of the three proposals, but it is still not perfect. The Sharpe 
Ratio looks at both return and risk, so it is better than looking only at return 
or only at volatility.

However, the Sharpe Ratio still looks at each asset separately. It does not fully 
consider whether the asset helps diversify the portfolio. An asset with a low 
Sharpe Ratio may still be useful if it has low or negative correlation with the 
other assets.

A better approach would be to test what happens to the whole portfolio when each 
asset is removed. The asset should be removed only if removing it improves the 
portfolio's overall risk-return trade-off.
"""
print(q3_answer)

#%%
