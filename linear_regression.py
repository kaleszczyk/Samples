
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


# In[15]:


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True) #dataset = datasets.load_diabetes(return_X_y=False)


# In[37]:


dataset = datasets.load_diabetes(return_X_y=False)
feature_names = dataset['feature_names']


# In[21]:


plt.scatter(diabetes_X[1:10,2], diabetes_y[1:10])


# In[56]:


row_count = 5
col_count = 2

training_data_lenght = 100
traning_range = range(0, training_data_lenght-1)

test_data_range = range(training_data_lenght, diabetes_X.shape[0])
model = linear_model.LinearRegression()

plt.figure(figsize=(20,20))

for i in range(row_count * col_count):
    plt.subplot(row_count, col_count, i+1)
    
    X_training = diabetes_X[traning_range, np.newaxis, i]
    y_training = diabetes_y[traning_range]
    
    model.fit(X_training, y_training)
    y_training_predicted = model.predict(X_training)
    print(model.coef_)
    
    plt.scatter(X_training, y_training)
    plt.plot(X_training, y_training_predicted, color='black')
    plt.title(feature_names[i], fontsize=20)
    
    X_test = diabetes_X[test_data_range, np.newaxis, i]
    y_test = diabetes_y[test_data_range]
    
    y_test_predicted = model.predict(X_test)
    print(model.coef_)
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_test_predicted, color='red', linewidth=0.5, linestyle='dashed')
    plt.title(feature_names[i], fontsize=20)
    
plt.show()

