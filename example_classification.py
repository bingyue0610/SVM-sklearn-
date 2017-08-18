#coding: utf-8

from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
print clf.fit(X, y)
print clf.predict([[2., 2.]])
# get support vectors
print clf.support_vectors_
# get indices of support vectors
print clf.support_
# get number of support vectors for each class
print clf.n_support_

#multi-class classification
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
print clf.fit(X, Y)

dec = clf.decision_function([[1]])
print dec.shape[1] # 4 classes: 4*3/2 = 6

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
print dec.shape[1] # 4 classes

#linearSVC implements “one-vs-the-rest” multi-class strategy
lin_clf = svm.LinearSVC()
print lin_clf.fit(X, Y)

dec = lin_clf.decision_function([[1]])
print dec.shape[1]