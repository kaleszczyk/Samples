
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[85]:


diabetes = pd.read_csv('datasets/diabetes.csv')


# In[86]:


diabetes.info()


# In[87]:


diabetes.groupby('Outcome').size()


# In[88]:


diabetes.hist(figsize=(10,10))


# In[89]:


diabetes.isnull().sum()
diabetes.isna().sum()


# In[90]:


diabetes_modified = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]


# In[91]:


diabetes_modified.info()


# In[92]:


from pandas.plotting import scatter_matrix
features = ["Age", "BMI", "Glucose"]
scatter_matrix(diabetes_modified[features], figsize=(12, 8))


# In[93]:


diabetes_modified.plot(kind = "scatter", x="BMI", y="Age" ,alpha = 0.1)


# In[94]:


diabetes_modified.plot(kind = "scatter", x="BMI", y="Age" ,alpha = 0.1, cmap=plt.get_cmap('summer'), c = diabetes_modified.Insulin, s=diabetes_modified.Insulin) 


# In[95]:


diabetes_modified.corr()


# In[96]:


from sklearn import preprocessing 

BMI_scaled = preprocessing.scale(diabetes_modified.BMI)


# In[97]:


BMI_scaled.mean()


# In[98]:


BMI_scaled.std()


# In[99]:


inputt = diabetes_modified.BMI.to_numpy()[:, np.newaxis]

#scaler  = preprocessing.StandardScaler().fit(inputt)

