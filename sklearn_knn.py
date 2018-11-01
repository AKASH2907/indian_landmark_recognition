import numpy as np
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier
import seaborn as sns
from sklearn import neighbors, datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors



# iris = datasets.load_iris()
# X = iris.data # we only take the first two features. 
# Y = iris.target

# print(X.shape)
# print(Y.shape)
# print(Y)

x_train = np.load('X_tr_irv2.npy')
x_valid = np.load('X_val_irv2.npy')
x_test = np.load('X_te_irv2.npy')
y_train = np.load('Y_train.npy')
y_val = np.load('Y_valid.npy')
y_test = np.load('Y_test.npy')

print(x_train.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[2]))
print(x_train.shape)
print(x_valid.shape)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
print(x_test.shape)


ground = []
test = []
for i in range(y_train.shape[0]):
	ground.append(np.argmax(y_train[i]))

for i in range(y_test.shape[0]):
	test.append(np.argmax(y_test[i]))

# print(ground)

clf = RandomForestClassifier(n_estimators=4, max_depth=2, random_state=0)
# print(clf)
clf.fit(x_train, ground)

# print(clf.feature_importances_)
rf = clf.predict(x_test)
print(rf)
acc_rf = accuracy_score(test, rf)
print(acc_rf)
cm_rf = confusion_matrix(test, rf)
print(cm_rf)

knn = KNeighborsClassifier(n_neighbors = 20).fit(x_train, ground) 
# accuracy = knn.score(x_test, test) 
y_pred = knn.predict(x_test)
acc = accuracy_score(test, y_pred)
print(acc)

# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, ground) 
# svm_predictions = svm_model_linear.predict(x_test)

# print(svm_predictions)

# accuracy = svm_model_linear.score(x_test, test) 
# print(accuracy)
knn_predictions = knn.predict(x_test)  
cm = confusion_matrix(test, knn_predictions) 

print(cm)

# nn = NearestNeighbors(n_neighbors=20)
