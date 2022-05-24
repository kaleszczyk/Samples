#załadowanie danych mnist
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

mnist = load_digits()
X= mnist["data"]
y=mnist["target"]
print(X.shape)
print(y.shape)

#wyświetlenie przykładowego obrazka 
import matplotlib 
import matplotlib.pyplot as plt

some_digit = X[1]
some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show

#podział danych na zbiory treningowy i testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#przetasowanie danych 
import numpy as np 
shuffle_index = np.random.permutation(1437)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#klasyfikator binarny (wykrywacz jednej cyfry tj 5 (dwie klasy "piątka" i "niepiątka"))
#wektory wyjściowe dla tego zadania: 
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)


#SGDClassifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

#walidacja krzyżowa
from sklearn.model_selection import cross_val_score

cgd_clf_score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#cross_val_predict - funkcja, która robi to samo co cross_val_score, tylko zamiast scora zwraca predykcje
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#macierz pomyłek
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_train_5, y_train_pred)

from sklearn.metrics import precision_score, recall_score 

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)