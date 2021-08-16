#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[22]:


iris=pd.read_csv(r'C:\Users\harsh\Documents\Python_Data_Sets\Iris.csv')
print(iris)


# In[23]:


print(iris.shape)


# In[24]:


print(iris.describe())


# In[25]:


#checking_for_null_values
print(iris.isna().sum())
print(iris.describe())


# In[26]:


iris.head()


# In[27]:


iris.head(150)


# In[28]:


iris.tail(100)


# In[29]:


n=len(iris[iris['Species']=='Iris-versicolor'])
print("No. of versicolor in dataset:",n)


# In[30]:


n=len(iris[iris['Species']=='Iris-virginica'])
print("No. of virginica in dataset:",n)


# In[31]:


n=len(iris[iris['Species']=='Iris-setosa'])
print("No. of setosa in dataset:",n)


# In[32]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l=('Versicolor','setosa','virginica')
s=[50,50,50]
ax.pie(s,labels=l,autopct='%1.2f%%')
plt.show()


# In[33]:


#checking_for_outliers
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['SepalLengthCm']])
plt.figure(2)
plt.boxplot([iris['SepalWidthCm']])
plt.show()


# In[34]:


iris.hist()
plt.show()


# In[36]:


iris.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)


# In[37]:


iris.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)


# In[39]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)


# In[40]:


sns.pairplot(iris,hue='Species')


# In[42]:


#Heatmap
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(iris.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[43]:


X=iris['SepalLengthCm'].values.reshape(-1,1)
print(X)


# In[44]:


Y = iris['SepalWidthCm'].values.reshape(-1,1)
print(Y)


# In[45]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()


# In[46]:


#Correlation 
corr_mat = iris.corr()
print(corr_mat)


# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[48]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[49]:


train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
train_y = train.Species

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
test_y = test.Species


# In[50]:


train_X.head()


# In[51]:


test_y.head()


# In[52]:


test_y.head()


# In[53]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[54]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[55]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[56]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[57]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# In[58]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# In[59]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)

