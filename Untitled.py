#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Linear Function

# In[2]:


x=np.linspace (-10,22)
plt.plot(x, x)
plt.title('Activation Function:Linear')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ### Sigmoid Function

# In[3]:


x=np.linspace (-10,22)
plt.plot(x, 1/(1+np.exp(-x)))
plt.title('Activation Function:Sigmoid')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ### Tanh Function

# In[4]:


x=np.linspace (-10,22)
plt.plot(x, np.tanh(x))
plt.title('Activation Function:Tanh')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ### Softmax Function 

# In[5]:


x=np.linspace (-10,20)
plt.plot(x,np.exp(x) / np.sum(np.exp(x),axis=0))
plt.title('Activation Function:Softmax')
plt.show()


# ### Relu Function

# In[6]:


x1=[]
for i in x:
    if i<0:
        x1.append(0)
    else:
        x1.append(i)

x=np.linspace (-10,20)
plt.plot(x,x1)
plt.title('Activation Function:RELU')
plt.show()

