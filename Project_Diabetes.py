

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB as BNB



diabetes = pd.read_csv("e:/Dell PC/Downloads/Online Courses/Projects/All-Hyperparamter-Optimization-master/diabetes1.csv")


diabetes["BloodPressure"] = np.where(diabetes["BloodPressure"] == 0, diabetes["BloodPressure"].median(), diabetes["BloodPressure"])
diabetes["Insulin"] = np.where(diabetes["Insulin"] == 0, diabetes["Insulin"].median(), diabetes["Insulin"])



#Splitting data


X = diabetes.iloc[:,:8]
y = diabetes.iloc[:,8]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


################ Naive Bayes Model #####################


model_BNB = BNB().fit(x_train, y_train)


confusion_matrix(model_BNB.predict(x_test), y_test)
test_accuracy = accuracy_score(model_BNB.predict(x_test), y_test)

confusion_matrix(model_BNB.predict(x_train), y_train)
train_accuracy = accuracy_score(model_BNB.predict(x_train), y_train)




################ Decision Tree Model #####################


model_DT = DT(criterion = 'entropy').fit(x_train, y_train)


confusion_matrix(model_DT.predict(x_test), y_test)
test_accuracy = accuracy_score(model_DT.predict(x_test), y_test)

confusion_matrix(model_DT.predict(x_train), y_train)
train_accuracy = accuracy_score(model_DT.predict(x_train), y_train)




################ Random Forest Model #####################


model_RF = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2, min_samples_leaf=1, n_jobs=1, random_state = 42)
model_RF.fit(x_train, y_train)


confusion_matrix(model_RF.predict(x_test), y_test)
test_accuracy = accuracy_score(model_RF.predict(x_test), y_test)

confusion_matrix(model_RF.predict(x_train), y_train)
train_accuracy = accuracy_score(model_RF.predict(x_train), y_train)




################ RandomForest Model using RandomizedSearchCV  #####################


rf = RandomForestClassifier()

RF_grid = {'n_estimators' : [100,200,300,400,500], 'max_features' : ['auto', 'sqrt', 'log2'], 'max_depth' : [int(x) for x in np.linspace(10, 1000,10)],'criterion': ['entropy','gini'], 'min_samples_split': [1,5,6,8], 'min_samples_leaf':[1,3,4,5]}


model_RF_Random = RandomizedSearchCV(rf, RF_grid, n_iter=100,cv=3,verbose=2, random_state=100, n_jobs=-1).fit(x_train, y_train)


confusion_matrix(model_RF_Random.predict(x_test), y_test)
test_accuracy = accuracy_score(model_RF_Random.predict(x_test), y_test)

confusion_matrix(model_RF_Random.predict(x_train), y_train)
train_accuracy = accuracy_score(model_RF_Random.predict(x_train), y_train)


model_RF_Random.best_params_



################ RandomForest Model using GridSearchCV #####################



parameters = {'criterion': ['gini'], 'max_depth': [450], 'max_features': ['log2'], 'min_samples_leaf': [1, 3, 5], 'min_samples_split': [3, 4, 5, 6, 7], 'n_estimators': [0, 100, 200, 300, 400]}

model_RF_grid = GridSearchCV(rf, parameters,scoring = 'accuracy', cv = 5).fit(x_train, y_train)

model_RF_grid.best_params_


confusion_matrix(model_RF_grid.predict(x_test), y_test)
test_accuracy = accuracy_score(model_RF_grid.predict(x_test), y_test)

confusion_matrix(model_RF_grid.predict(x_train), y_train)
train_accuracy = accuracy_score(model_RF_grid.predict(x_train), y_train)










