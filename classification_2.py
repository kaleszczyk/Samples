
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[16]:


neightbors_count = 5

classifier = KNeighborsClassifier(n_neighbors=neightbors_count)

rnd_state = 2 #seed do losowania liczb zeby kazdy mial to samo wylosowane 

X, y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=rnd_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rnd_state)

plt.subplot(1, 4, 1)
plt.scatter(X[:, 0],X[:, 1], c=y)
plt.title("Whole dataset")

plt.subplot(1, 4, 2)
plt.scatter(X_train[:, 0],X_train[:, 1], c=y_train)
plt.title("Training")

plt.subplot(1, 4, 3)
plt.scatter(X_test[:, 0],X_test[:, 1], c=y_test)
plt.title("Test")

classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)

plt.subplot(1, 4, 4)
plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction)
plt.title("Prediction")


# In[18]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# In[26]:


from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

y_score = classifier.predict_proba(X_test)
print(y_score)
fpr, tpr, treshold = roc_curve(y_test, y_score[:,1])
print(fpr)
print(tpr)
print(treshold)

plt.step(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")


# In[38]:


precisions, recalls, tresholds = precision_recall_curve(y_test, y_score[:,1])

def plot_precision_recall_vs_treshold(precisions, recalls, thresholds):
    plt.plot(tresholds, precisions[:-1], "r--", label="Precision")
    plt.plot(tresholds, recalls[:-1], "k--", label="Recall")
    plt.xlabel("Treshold")
    plt.legend(loc="best")
    plt.ylim([0,1.1])
    

plot_precision_recall_vs_treshold(precisions, recalls, tresholds)

