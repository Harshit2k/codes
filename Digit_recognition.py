#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


digit_test=pd.read_csv(r'C:\Users\harsh\Documents\Python_Data_Sets\digit-recognizer\test.csv')
digit_train=pd.read_csv(r'C:\Users\harsh\Documents\Python_Data_Sets\digit-recognizer\train.csv')


# In[4]:


digit_train.head()


# In[5]:


digit_train.shape


# In[6]:


digit_test.shape


# In[7]:


x=digit_train.drop(['label'],axis=1).values
y=digit_train['label'].values


# In[8]:


print(y[40])


# In[9]:


d=x[40].reshape(28,28)


# In[10]:


import matplotlib.pyplot as plt
plt.imshow(d)
plt.show()


# In[11]:


digit_test=digit_test.values
a=digit_test[35].reshape(28,28)
plt.imshow(a)
plt.show()


# In[ ]:




