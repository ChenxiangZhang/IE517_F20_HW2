import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

#import Iris dataset
df=pd.read_csv('D:/UIUC/IE517 machine learning/week2/Treasury Squeeze test - DS1.csv')
X=df.iloc[:,2:11]
y=df.iloc[:,11]

#split train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33,stratify=y)

# Standardizing the features:
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#decisiontreeclassifiers with  different criterion
# dt=DecisionTreeClassifier(criterion='gini',random_state=33)
# dt.fit(X_train,y_train)
# y_pred=dt.predict(X_test)
# print(accuracy_score(y_test,y_pred))

dt1=DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state=33)
dt1.fit(X_train,y_train)
y_pred1=dt1.predict(X_test)
# print(accuracy_score(y_test,y_pred1))

dot = export_graphviz(dt1,
                           filled=True, 
                           rounded=True,
                           class_names=['True', 
                                        'False'],
                           feature_names=['price_crossing',
                                          'price_distortion',
                                          'roll_start',
                                          'roll_heart',
                                          'near_minus_next',
                                          'ctd_last_first',
                                          'ctd1_percent',
                                          'delivery_cost',
                                          'delivery_ratio'],
                           out_file=None) 
graph = graph_from_dot_data(dot) 
graph.write_png('tree.png')


from PIL import Image
im = Image.open('tree.png')
im.show()



#KNN Training
k_range=range(1,50)
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

plt.plot(range(1,50),score, label='testing accuracy')
plt.plot(range(1,50),score1,label='training accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()  
print("My name is Chenxiang Zhang and you can call me Franklin")
print("My NetID is:cz52")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

