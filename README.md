# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input
   
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sabeeha Shaik
Register Number:  212223230176
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
![image](https://github.com/user-attachments/assets/d052c001-82cd-432e-a890-a38ce347f03d)

```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/bcc92f8e-74aa-4b86-b27b-03cac284cf8c)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/ca76166a-7fa7-4d57-9b01-6f3a8b84d8af)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
![image](https://github.com/user-attachments/assets/371e42a7-665d-4c35-9804-c75613853cfe)

```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```
![image](https://github.com/user-attachments/assets/cd0a4dd6-0a7b-4837-80d6-83701b97442d)

```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/85b20974-c564-459e-897c-b55d16eec77a)
```

print(Y)
```

![image](https://github.com/user-attachments/assets/33ccfc2d-808f-44f9-b53b-7a826aa4f266)

```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/a561406e-a89f-4664-847c-2a318c10120d)
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
print("Sabeeha Shaik")
print(212223230176)
```
![image](https://github.com/user-attachments/assets/c1e09f5c-0dc7-4f69-a12b-e1fdb48b6539)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

