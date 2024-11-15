# Ex-7:SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data

2. Split Dataset into Training and Testing Sets

3. Train the Model Using Stochastic Gradient Descent (SGD)

4. Make Predictions and Evaluate Accuracy

5. Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Suriya Pravin M
RegisterNumber:  212223230223

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']= iris.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train,y_train)

y_pred =sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test, y_pred)
print("confusion Matrix:")
print(cm)
*/
```

## Output:
### Iris dataset

![1w](https://github.com/user-attachments/assets/0d227501-0732-4a86-9835-cd83dea8379a)

### Accuracy
![ds](https://github.com/user-attachments/assets/f1946dd3-6617-40c2-910f-8d0eda3b3501)


### Confusion Matrix
![s](https://github.com/user-attachments/assets/479d2654-80ac-45d5-bdee-fd8409445f42)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
