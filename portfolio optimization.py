#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize


# In[3]:


tickers = ['SPY','BND','GLD','QQQ','VTI']


# In[5]:


endDate = datetime.today()


# In[9]:


start_date = endDate - timedelta(days=5*365)
print(start_date)


# In[13]:


Adj_close_df = pd.DataFrame()


# In[15]:


for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=endDate)
    Adj_close_df[ticker]=data['Adj Close']


# In[19]:


print(Adj_close_df)


# In[23]:


log_returns=np.log(Adj_close_df/Adj_close_df.shift(1))


# In[25]:


log_returns=log_returns.dropna()


# In[29]:


cov_matrix=log_returns.cov()*252
print(cov_matrix)


# In[31]:


def standard_deviation (weights, cov_matrix):
    variance=weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


# In[103]:


def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252


# In[121]:


def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return (weights, log_returns)- risk_free_rate)/standard_deviation(weights, cov_matrix)


# In[123]:


risk_free_rate=0.02


# In[125]:


def neg_sharpe_ratio(weights,log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)


# In[127]:


costraints= {'type':'eq','fun':lambda weights: np.sum(weights)-1}
bounds= [(0, 0.5) for _ in range(len(tickers))]


# In[129]:


initial_weights= np.array([1/len(tickers)]*len(tickers))
print(initial_weights)


# In[140]:


optimized_results = minimize(neg_sharpe_ratio, initial_weights,args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP')


# In[142]:


optimal_weights = optimized_results.x


# In[154]:


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


# In[158]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()

