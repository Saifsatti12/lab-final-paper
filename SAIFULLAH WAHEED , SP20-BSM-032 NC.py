#!/usr/bin/env python
# coding: utf-8

# In[2]:


#qno:1 (1)


# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


a1 = np.linspace (0,10, 100)


# In[6]:


a1


# In[7]:


# Qno:1 (2)


# In[41]:


np.arange(0,100,3)


# In[11]:


#Qno:1 (3)


# In[10]:


x=np.arange(1,10,1)
y=x**3-5*x-9
plt.plot(x,y,"o:")
plt.xlabel("x_axis")
plt.ylabel("y_axis")
plt.grid()
plt.show()


# In[12]:


#Qno:1 (6)


# In[18]:


list = ['Karachi','Lahore','Multan','Rawalpindi','Islamabad']


# In[19]:


list.append ('Faisalabad')


# In[20]:


list.append ('Kohat')


# In[21]:


list.append ('Murree')


# In[22]:


list.append('Peshawar') 


# In[24]:


list.append ('Sargodah')


# In[25]:


print(list)


# In[42]:


#another method


# In[45]:


list = ['Karachi','Lahore','Multan','Rawalpindi','Islamabad']
list.extend(["Kasur","Sahiwal", "Quetta", "Peshawar", "Attock"]) 
print(list)


# In[26]:


#Qno:1 (5)


# In[48]:


from math import sin
x=input("Enter a float:")

def f(x):
    x**2 - sin(x)**2 - 4*x + 1
print("value of x: ", x)


# In[33]:


f = lambda x:2*x**3-9.5*x+7.5


# In[ ]:





# In[ ]:


#Qno:05


# In[46]:


rom math import sin
def bisection(x0,x1,e): 
    step = 1
    condition = True
    while condition:
        x2 = (x0+x1)/2
        print('iteration %d, x2 = %0.6f and f(x2)= %0.6f' %(step,x2,f(x2)))
        
        if f(x0) * f(x2) < 0:
            x1 = x2
        else:
            x0 = x2
        step = step +1
        condition = abs(f(x2)) > e
    print('root is :%0.8f '%x2)
#    return x2

def f(x):
    return x**3-5*x-9

x0 = float(input('first guess: '))
x1 = float(input('second guess: '))
e  = float(input('tolerance: '))

if f(x0) * f(x1) > 0.0:
    print('given guess values do not bracket the root')
else:
    root = bisection(x0,x1,e)2


# In[35]:


#Qno:2


# In[37]:


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

get_ipython().run_line_magic('matplotlib', 'inline')
def divided_diff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] =            (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
     '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p
x = np.array([-5, -1, 0, 2])
y = np.array([-2, 6, 1, 3])
# get the divided difference coef
a_s = divided_diff(x, y)[0, :]

# evaluate on new data points
x_new = np.arange(-5, 2.1, .1)
y_new = newton_poly(a_s, x, x_new)

plt.figure(figsize = (12, 8))
plt.plot(x, y, 'bo')
plt.plot(x_new, y_new)


# In[38]:


#Qno:3


# In[39]:


import numpy as np
array= np.arange(1,30)
print("array of the integersfrom 30 to 70")
print(array)


# In[47]:


#Qno:4


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import interpolate


# In[50]:


a1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,])


# In[51]:


a2 = np.linspace(0,10, 100)


# In[52]:


np.arange(0,100,3)


# In[53]:


a2


# In[54]:


a3 = np.linspace(1, 100)


# In[55]:


a3 = np.linspace(1, 100)


# In[56]:


a3-3


# In[ ]:




