# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

Step1:
Import pandas as pd.

Step2:
Read the csv file.

Step3:
Get the value of X and y variables.

Step4:
Create the linear regression model and fit.

Step5:
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3.

Step6:
Print the predicted output.

## Program:
```
'''
Developed By
             Name:S.Meena
             Ref No:21500895
'''             
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
x1=pd.read_csv('clustering.csv')
print(x1.head(2))
x2=x1.loc[:,['ApplicantIncome','LoanAmount']]
print(x2.head(2))

x=x2.values
#print(x)
sns.scatterplot(x[:,0],x[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()

kmeans=KMeans(n_clusters=4)
kmeans.fit(x)
print("Cluster centers:",kmeans.cluster_centers_)
print("Labels:",kmeans.labels_)
predict_class=kmeans.predict([[9000,1200]])
print("Cluster group for application income 9000 and loanamount 120 is:",predict_class)





```
## Output:

![ML1](https://user-images.githubusercontent.com/94677128/154296333-0e291b54-a830-4d0a-9202-832cbfc66d6e.png)


![ML2](https://user-images.githubusercontent.com/94677128/154296503-8a147461-0ca3-4c48-a52c-4dc2761c79c2.png)


## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program.
