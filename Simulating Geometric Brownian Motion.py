import numpy as np
import matplotlib.pyplot as plt

mu= 0.1
n= 100
T= 1
M= 100
S0=100
sigma=0.3

dt=T/n
St=np.exp(
    (mu-sigma**2/2)*dt
    +sigma*np.random.normal(0,np.sqrt(dt),size=(M,n)).T
    )

St = np.vstack([np.ones(M), St])

St = S0*St.cumprod(axis=0)

time=np.linspace(0,T,n+1)

tt=np.full(shape=(M,n+1), fill_value=time).T

plt.plot(tt, St)
plt.xlabel('Years $(t)$')
plt.ylabel('Stock Price $(S_t)$')
plt.title('Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n')
         
plt.show()
