
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs


# In[16]:


plt.figure(figsize=(20,20))
plt.subplot(3, 2, 1)
X, y = make_classification(n_features=2, n_informative=1, n_clusters_per_class=1, n_redundant=0)
#X-wspołrzedne
#y-etykiety
plt.scatter(X[:,0], X[:, 1], c=y) 

plt.subplot(3, 2, 2)
X, y = make_classification(n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0)
#X-wspołrzedne
#y-etykiety
plt.scatter(X[:,0], X[:, 1], c=y) 

plt.subplot(3, 2, 3)
X, y = make_classification(n_features=2, n_informative=2, n_clusters_per_class=2, n_redundant=0)
#X-wspołrzedne
#y-etykiety
plt.scatter(X[:,0], X[:, 1], c=y) 

plt.subplot(3, 2, 4)
X, y = make_classification(n_features=2, n_informative=2, n_clusters_per_class=2, n_redundant=0, n_classes=2)
#X-wspołrzedne
#y-etykiety
plt.scatter(X[:,0], X[:, 1], c=y) 

plt.subplot(3, 2, 5)
X, y = make_blobs(n_features=2, centers=3)
plt.scatter(X[:,0], X[:, 1], c=y) 

plt.subplot(3, 2, 6)
X, y = make_blobs(n_features=3, centers=3)
plt.scatter(X[:,0], X[:, 1], c=y) 

