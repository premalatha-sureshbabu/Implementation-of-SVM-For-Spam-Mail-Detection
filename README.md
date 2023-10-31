# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: S.Prema Latha
RegisterNumber:  212222230112

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/2721361c-4c33-42d2-badd-38a4e2739ad5)
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/051bd5f5-77cc-488d-8384-b0292298c98d)
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/8f64e653-6637-447f-a447-d2e6dfd4012c)

![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/94d9ad0d-c7df-4d8a-9a95-92cd5ea1b5b4)
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/d1013d20-b5cd-4652-ab4d-7ee9cfac7780)
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/e2e9671e-5993-4ef8-9319-13829fb3a9f1)
![image](https://github.com/premalatha-sureshbabu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120620842/e01ad6f7-1834-4b10-8edd-bd3fdb829ae3)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
