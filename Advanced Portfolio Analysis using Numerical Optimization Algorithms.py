
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

tickers = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]

data = yf.download(tickers.Symbol.to_list(),'2021-01-01')['Adj Close']

returns = data.pct_change().dropna()

expected_returns = returns.mean()*252
cov_matrix = returns.cov()*252

def portfolio_variance(w):
    return (w.dot(cov_matrix)).dot(w)

n_assets = len(data.columns)
equal_weight=1/n_assets

initial_weights = np.array([1/n_assets for i in range(n_assets)])
initial_weights

plt.plot(np.linspace(0,10,3))

target_returns=np.linspace(expected_returns.min(), expected_returns.max(), 50)

target_vols=[]
for target_return in target_returns:
    constraints=({'type':'eq','fun': lambda x:np.sum(x)-1},
               {'type':'eq','fun': lambda x: x.dot(expected_returns)-target_return})
    bounds= [[0,1]]*n_assets
    result= minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    target_vols.append(result.fun**(1/2))

plt.scatter(target_vols, target_returns, label='EF')
plt.xlabel('Volatility')
plt.ylabel('Expected returns')
plt.title('Efficient frontier for the Dow Jones Industrial Avarage')
plt.show

