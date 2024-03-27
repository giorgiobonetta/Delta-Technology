#!/usr/bin/env python
# coding: utf-8

# Value at Risk
# The Value at Risk is the answer of this question: Which is the maximum loss that could occur for a given probability, in a certain period of time?
# So, it is possible to conduct a stress test estimating the maximum loss of a diversified portfolio.

# Let's start importing a few library

# In[3]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


# Now, we go head selecting the time period as the VaR depends on interval time. For example, we selected an interval form 15 years age to now. 

# In[5]:


year = 15

endDate = dt.datetime.now()
startDate = endDate-dt.timedelta(days= 365 * year)

There, 4 tickers of different assets have been picked: 
> SPDR S&P 500 ETF
> Vanguard Total Bond Market ETF
> Invesco Trust Series 1
> Vanguard Total Stock Market ETF
# In[7]:


tickers = ['SPY','BND','QQQ','VTI']

Down below, throughout a for cycle, we load the Adjusted closing prices, which are the prices after adjustments for all applicable splits and dividend distributions. 
Data is adjusted using appropriate split and dividend multipliers, adhering to Center for Research in Security Prices (CRSP) standards.
I leave there the Yahoo Finance page for further information
elp.yahoo.com/kb/SLN28256.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLml0Lw&guce_referrer_sig=AQAAAB_OaNFWWrsqyw5QeEpUPw1M8JjPmwTdHUDDQmo-Su2ae9EXYbmLrAOeOBariwxbQdByaSwarRUBUydgxAO4XCgnHicBnSqqkzdNXQIIvPLYoQNUHU75rIERV37caR8VREBpA6U-GJAfI4PURC_zfC98ZEyTLiMaHHAZrGytOa8g
# In[26]:


adj_close_df= pd.DataFrame()
for ticker in tickers:
    data= yf.download(ticker, start=startDate, end=endDate)
    adj_close_df[ticker]= data['Adj Close']

print(adj_close_df)


# In[30]:


log_returns=np.log(adj_close_df/adj_close_df.shift(1))
log_returns=log_returns.dropna()

print(log_returns)


# In[32]:


portfolio_value=1000000
weights=np.array([1/len(tickers)]*len(tickers))
print(weights)


# In[34]:


historical_returns=(log_returns*weights).sum(axis=1)
print(historical_returns)


# In[36]:


days=5
range_returns=historical_returns.rolling(window=days).sum()
range_returns=range_returns.dropna()
print(range_returns)


# In[38]:


confidence_interval= 0.95
VaR=-np.percentile(range_returns, 100 - (confidence_interval*100))*portfolio_value
print(VaR)


# In[49]:


return_window=days
range_returns=historical_returns.rolling(window=return_window).sum()
range_returns=range_returns.dropna()

range_returns_dollar=range_returns*portfolio_value

plt.hist(range_returns_dollar.dropna(), bins=50, density=True)
plt.xlabel(f'{return_window}-Day Portfolio Return (Dollar Value)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio {return_window} - Day Returns (Dollar value)')
plt.axvline(-VaR, color="r", linestyle="dashed", linewidth=2, label=f'Var at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()


# In[ ]:




