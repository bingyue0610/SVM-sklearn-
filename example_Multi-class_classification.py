#coding:utf-8

'''
SVC and NuSVC implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification.
If n_class is the number of classes, then n_class * (n_class - 1) / 2 classifiers are constructed and each one
 trains data from two classes. To provide a consistent interface with other classifiers, the decision_function_shape
  option allows to aggregate the results of the “one-against-one” classifiers to a decision function of shape
  (n_samples, n_classes)

On the other hand, LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_class models.
If there are only two classes, only one model is trained:

Note that the LinearSVC also implements an alternative multi-class strategy, the so-called multi-class SVM
formulated by Crammer and Singer, by using the option multi_class='crammer_singer'. This method is consistent,
which is not true for one-vs-rest classification. In practice, one-vs-rest classification is usually preferred,
since the results are mostly similar, but the runtime is significantly less.

see more:http://scikit-learn.org/stable/modules/svm.html
'''

import numpy as np
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
dec = lin_clf.decision_function([[1]])
dec.shape[1]
