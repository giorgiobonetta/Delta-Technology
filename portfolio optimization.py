#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

tickers = ['SPY','BND','GLD','QQQ','VTI']

endDate = datetime.today()
start_date = endDate - timedelta(days=5*365)
print(start_date)


Adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=endDate)
    Adj_close_df[ticker]=data['Adj Close']

print(Adj_close_df)

log_returns=np.log(Adj_close_df/Adj_close_df.shift(1))
log_returns=log_returns.dropna()

cov_matrix=log_returns.cov()*252
print(cov_matrix)

def standard_deviation (weights, cov_matrix):
    variance=weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return (weights, log_returns)- risk_free_rate)/standard_deviation(weights, cov_matrix)

risk_free_rate=0.02

def neg_sharpe_ratio(weights,log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
    
costraints= {'type':'eq','fun':lambda weights: np.sum(weights)-1}
bounds= [(0, 0.5) for _ in range(len(tickers))]

initial_weights= np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

optimized_results = minimize(neg_sharpe_ratio, initial_weights,args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP')

optimal_weights = optimized_results.x

print('optimal_weights:')
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}:{weight:.4f}")
    
print()

optimal_portfolio_return=expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility= standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio=sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return:{optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}:")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()

