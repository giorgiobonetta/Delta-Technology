#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install streamlit


# In[9]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# In[11]:


st.title('Investment Portfolio Dashboard')
assets= st.text_input('Provide your assets (comma-separated)','AAPL,MSFT,GOOGL')
start= st.date_input('Pick a starting date for your analysis', value=pd.to_datetime('2022-06-01'))
data=yf.download(assets,start=start)['Adj Close']


# In[ ]:





# In[15]:


ret_df=data.pct_change()
cumul_ret=(ret_df+1).cumprod()-1
pf_cumul_ret=cumul_ret.mean(axis=1)


# In[20]:


benchmark=yf.download('^GSPC',start=start)['Adj Close']
bench_ret=benchmark.pct_change()
bench_dev=(bench_ret+1).cumprod()-1


# In[22]:


w=(np.ones(len(ret_df.cov()))/len(ret_df.cov()))
pf_std=(w.dot(ret_df.cov()).dot(w))**(1/2)


# In[24]:


st.subheader('Portfolio vs. Index Development')
tog=pd.concat([bench_dev, pf_cumul_ret],axis=1)
tog.columns=['S&P500 Performance','Portfolio Performance']


# In[26]:


st.line_chart(data=tog)


# In[30]:


st.subheader('Portfolio Risk:')
pf_std

st.subheader('Benchmark Risk:')
bench_risk=bench_ret.std()
bench_risk


# In[34]:


st.subheader('Portfolio competition:')
fig, ax= plt.subplots(facecolor='#121212')
ax.pie (w,labels=data.columns, autopct='%1.1f%%', textprops={'color':'white'})

st.pyplot(fig)


# In[ ]:




