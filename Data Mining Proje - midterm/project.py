import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


data = pd.read_csv('data.csv')

##step 1 (traning data and clasification)
X = data.iloc[:,:-1] 
y = data.iloc[:,-1]  #last coloumn selected for the y (target)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=12)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
k = 20
metric_array = np.zeros((k-1))

for n in range(1,k):
    model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    y_hat = model.predict(X_test)
    metric_array[n-1] = metrics.accuracy_score(y_test,y_hat)


k=3
model_knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

#result
y_hat_knn = model_knn.predict(X_test)


### SVM classification
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_hat_svm = clf.predict(X_test)


#confusion matrix for SVM
all_labels = sorted(np.unique(y_test)) 
confusion_matrix_svm = confusion_matrix(y_test, y_hat_svm, labels=all_labels)
cm_display_svm = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_svm,
    display_labels=all_labels
)
cm_display_svm.plot()
plt.title('FIRST Confusion matrix for svm')
plt.show()


#confusion matrix for KNN
confusion_matrix_knn = confusion_matrix(y_test,y_hat_knn,labels=all_labels)
cm_display_knn = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_knn,
    display_labels=all_labels
)
cm_display_knn.plot()
plt.title('FIRST Confusion matrix for knn')
plt.show()



## 2. step (giving data randomly null values)
random_numbers = random.sample(range(17, 101), 5)
data['Age'] = data['Age'].replace([random_numbers],np.nan)
print('NaN data replaced by randomly values\n',data)


## 3. step (filling null datas with coloumn mean)
data = data.fillna(data['Age'].mean())
print('NaN values replaced by itself column mean',data)


## 4. step(training new data like step 1)
X = data.iloc[:,:-1] 
y = data.iloc[:,-1] #last coloumn selected for the y (target)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=12)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
k = 20
metric_array = np.zeros((k-1))

for n in range(1,k):
    #print(n)
    model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    y_hat = model.predict(X_test)
    metric_array[n-1] = metrics.accuracy_score(y_test,y_hat)

#print(metric_array) # with this we found the k = 3 is best for classification knn

k=3
model_knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

#result
y_hat_knn = model_knn.predict(X_test)


###svm classification
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_hat_svm = clf.predict(X_test)
#print('svm result' , y_hat_svm)


#confusion matrix for svm(new data)
all_labels = sorted(np.unique(y_test)) 
confusion_matrix_svm = confusion_matrix(y_test, y_hat_svm, labels=all_labels)
cm_display_svm = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_svm,
    display_labels=all_labels
)
cm_display_svm.plot()
plt.title('LAST Confusion matrix for svm')
plt.show()

#confusion matrix for knn(new data)
confusion_matrix_knn = confusion_matrix(y_test,y_hat_knn,labels=all_labels)
cm_display_knn = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_knn,
    display_labels=all_labels
)
cm_display_knn.plot()
plt.title('LAST Confusion matrix for knn')
plt.show()