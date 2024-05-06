# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Udhayanithi M
RegisterNumber: 212222220054
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

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
### Encoding
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/14c93d11-d8d5-4f91-8b80-1ba4f76af49d)

### Head()
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/49754a09-46dc-4838-8c17-7e6d9203f94a)

### Info()
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/3d140aa9-c186-4943-a70b-a56460a90375)

### isnull().sum()
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/995fe78e-4661-46d3-84e3-fada28820617)

### Prediction of y
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/3dc11444-a410-4950-a8d9-fde59ed078e9)

### Accuracy
![image](https://github.com/23004426/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144979327/7f0a5fc6-1c65-44be-9217-370588a7bf9a)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
