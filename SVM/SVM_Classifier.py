import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

#Getting data from sklearn dataset module
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)
x = cancer.data
y = cancer.target

#test,train split
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
# print(x_train,y_train)

classes = ['malignant' 'benign']

#Fitting into SVC,SVM model with kernal defined
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

#Predicting and accuracy
y_pred = clf.predict(x_test)
print(y_pred)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)
acc = clf.score(x_test,y_test)
print(acc)

