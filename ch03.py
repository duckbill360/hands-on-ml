# ch03
# Classification

# Download the MNIST dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='mnist_dataset/')


# This is dataset contains 70000 pictures and each of them has 784 features (28 x 28 pixels)
# Each pixel has an intensity from 0 to 255
X, y = mnist['data'], mnist['target']


# Split into train and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the sets
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# A good place to start is with a Stochastic Gradient Descent (SGD) classifier,
# using Scikit-Learn’s SGDClassifier class.
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)


# Now you can use it to detect images of the number 5:
some_digit = X[36000]
print(sgd_clf.predict(some_digit))
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)


# Fine-Tune the model
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_train, y_train_pred)

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# # Plot the confusion matrix
import matplotlib.pyplot as plt
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# Multi-Label Classification (p.100)
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))
