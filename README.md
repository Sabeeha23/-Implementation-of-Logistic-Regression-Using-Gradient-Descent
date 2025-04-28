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
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
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
```
print(y_pred)
```
```
print(Y)
```
```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```
```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
print("Sabeeha Shaik")
print(212223230176)
```
## Output:
## Dataset
![image](https://github.com/user-attachments/assets/95d4ce91-23c4-4a39-a663-209c721b573c)

## dtypes
![image](https://github.com/user-attachments/assets/d6e5c8ba-8653-4aac-b62d-4af60c430dd1)

## dataset
![image](https://github.com/user-attachments/assets/1227d5b9-f8c5-4f2b-82d7-64e950826da2)

## y array
![image](https://github.com/user-attachments/assets/012d737a-a7b4-46e7-aff2-d8d2cabd780b)

## Accuracy
![image](https://github.com/user-attachments/assets/54a09eaa-6038-47dc-998b-ae1e685d985b)

## y_pred
![image](https://github.com/user-attachments/assets/718f5c2f-c849-49cf-92c3-dc8bac3d24e4)

## y
![image](https://github.com/user-attachments/assets/4b8af230-1758-470f-b03e-2b5f2cc75652)

## y_prednew
![image](https://github.com/user-attachments/assets/9595b5d4-3312-4439-b90a-6397d7def82e)

## y_prednew
![image](https://github.com/user-attachments/assets/44510ec0-7262-4429-9235-af833c9709ea)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

