#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"C:\Users\HP\Downloads\House Prediction\train.csv")


# In[5]:


df.shape


# In[6]:


df.head(6)


# In[9]:


pd.set_option( 'display.max_columns',None)
pd.set_option( 'display.max_rows',None)


# In[10]:


df.head()


# In[11]:


df.info()


# In[13]:


df.isnull().sum()


# In[18]:


null_var = df.isnull().sum() / df.shape[0] *100
null_var


# In[19]:


drop_columns = null_var[null_var>17].keys()
drop_columns


# In[20]:


df2_drop_clm = df.drop(columns =drop_columns )


# In[23]:


df2_drop_clm.shape


# In[24]:


sns.heatmap(df2_drop_clm.isnull())


# In[25]:


df3_drop_rows = df2_drop_clm.dropna()


# In[26]:


df3_drop_rows.shape


# In[28]:


sns.heatmap(df3_drop_rows.isnull())


# In[30]:


df3_drop_rows.isnull().sum()


# In[31]:


df3_drop_rows.select_dtypes(include=['int64','float64']).columns


# In[32]:


sns.displot(df['MSSubClass'])


# In[33]:


sns.displot(df3_drop_rows['MSSubClass'])


# In[34]:


sns.displot(df['MSSubClass'])
sns.displot(df3_drop_rows['MSSubClass'])


# In[37]:


num_var = [ 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'SalePrice']
plt.figure(figsize = (25,25))
for i,var in enumerate(num_var):
    plt.subplot(9,4,i+1)
    sns.displot(df[var],bins=20)
    sns.displot(df3_drop_rows[var],bins=20)
    


# In[38]:


df3_drop_rows.select_dtypes(include=['object']).columns


# In[45]:


pd.concat([df['MSZoning'].value_counts() / df.shape[0]*100,
                 df3_drop_rows['MSZoning'].value_counts()/df3_drop_rows.shape[0]*100], axis =1,
                 keys=['MSZoning_org','MSZoning_clean'])


# In[48]:


def cat_var_dist(var):
    return pd.concat([df[var].value_counts() / df.shape[0]*100,
                 df3_drop_rows[var].value_counts()/df3_drop_rows.shape[0]*100], axis =1,
                 keys=[var +'org',var + 'clean'])


# In[49]:


cat_var_dist('MSZoning')


# In[ ]:




