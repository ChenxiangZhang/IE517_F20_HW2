from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#import Iris dataset
df=pd.read_csv('D:/Iris.csv')
X=df.iloc[:,:4]
y=df.iloc[:,4]

#plot_decision_regions
iris = datasets.load_iris()
X1 = iris.data[:, [2, 3]]
y1= iris.target

#split train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33,stratify=y)
#plot_decision_regions
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size=0.25,random_state=33,stratify=y1)

# Standardizing the features:
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#plot_decision_regions
scaler1=preprocessing.StandardScaler().fit(X_train1)
X_train1_std=scaler1.transform(X_train1)
X_test1_std=scaler1.transform(X_test1)


X_combined_std = np.vstack((X_train1_std, X_test1_std))
y_combined = np.hstack((y_train1, y_test1))

#decisiontreeclassifiers with  different criterion
# dt=DecisionTreeClassifier(criterion='gini',random_state=33)
# dt.fit(X_train,y_train)
# y_pred=dt.predict(X_test)
# print(accuracy_score(y_test,y_pred))

# dt1=DecisionTreeClassifier(criterion='entropy',random_state=33)
# dt1.fit(X_train,y_train)
# y_pred1=dt1.predict(X_test)
# print(accuracy_score(y_test,y_pred1))

#KNN Training
k_range=range(1,26)
score=[]
score1=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    y_train_pred=knn.predict(X_train)
    score.append(accuracy_score(y_test,y_pred))
    score1.append(accuracy_score(y_train,y_train_pred))
    
print('accuracy',max(score))
print('best K',score.index(max(score)))

plt.plot(range(1,26),score, label='testing accuracy')
plt.plot(range(1,26),score1,label='training accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()  



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

## Building a decision tree
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train1, y_train1)

X_combined = np.vstack((X_train1, X_test1))
y_combined = np.hstack((y_train1, y_test1))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()


# # K-nearest neighbors - a lazy learning algorithm
knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train1_std, y_train1)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()

print("My name is Chenxiang Zhang and you can call me Franklin")
print("My NetID is:cz52")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")