# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and Load the dataset.

2.Define X and Y array and Define a function for costFunction,cost and gradient.

3.Define a function to plot the decision boundary.

4.Define a function to predict the Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: A.Anbuselvam
RegisterNumber:  212222240009
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:
## Array value of x:
![Screenshot 2023-09-23 113924](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/0a2c952a-11b1-43a7-9784-33997de0ff6a)

## Array value of y:
![Screenshot 2023-09-23 113930](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/7e1ccd38-abcf-40d8-8488-17b4834bb54e)

## Score graph:
![Screenshot 2023-09-23 113945](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/14225e44-60c4-47d6-9cb0-bd31a1be04bf)

## Sigmoid function graph:
![Screenshot 2023-09-23 114000](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/cf87b517-a789-4e0f-9e5b-ef9ee3bdbc17)

## X train grad value:
![Screenshot 2023-09-23 114012](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/ec58877b-d461-4e90-86f7-57333b6f78db)

## Y train grad value:
![Screenshot 2023-09-23 114019](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/06db7e0d-81a2-4ffe-a1a9-0c210af561df)

## Regression value:
![Screenshot 2023-09-23 114031](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/074b9b47-6147-431b-b967-68289b8f6436)

## decision boundary graph:
![Screenshot 2023-09-23 114059](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/7555b35d-7bd5-4697-81bb-4b86ae07a9fc)

## Probability value:
![Screenshot 2023-09-23 114109](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/7cef2adf-edc1-410b-baaf-12500d9961c5)

## Prediction mean value:
![Screenshot 2023-09-23 114114](https://github.com/anbuselvamA/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559871/ef07ca3f-2d4d-46ce-a252-26122e15fe00)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

