
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df=pd.read_csv('logistic.csv')
df.head()


# In[3]:


X=df[['age']]
Y=df.insurance


# In[4]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=10)


# In[5]:


X_test


# In[6]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)


# In[7]:


model.predict(X_test)


# In[8]:


model.predict(25)


# In[9]:


model.score(X_test,Y_test)


# In[10]:


model.predict_proba(X_test)


# In[11]:


from matplotlib import pyplot as plt


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.age,df.insurance,marker='+',color='red')
plt.xlabel('age')
plt.ylabel('insurance')

