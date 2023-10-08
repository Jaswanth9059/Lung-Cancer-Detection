#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Classified Data")
df


# In[3]:


df.isnull().sum()


# In[4]:


pd.read_csv("Classified Data", index_col = 0)


# In[5]:


x = df.iloc[:,1:11].values
x


# In[6]:


y = df.iloc[:,11:12].values
y


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scaler = StandardScaler()


# In[9]:


x = scaler.fit_transform(x)


# In[10]:


x


# In[11]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .30, random_state = 8)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[77]:


knn = KNeighborsClassifier(n_neighbors = 7)


# In[78]:


knn.fit(xtrain, ytrain)


# In[79]:


pred = knn.predict(xtest)


# In[80]:


from sklearn.metrics import classification_report,confusion_matrix


# In[81]:


print(confusion_matrix(ytest,pred))


# In[82]:


from sklearn.metrics import accuracy_score


# In[83]:


a = accuracy_score(ytest,pred)
print(f"Accuracy : {a}")


# In[84]:


print(classification_report(ytest,pred))


# # Choosing K value

# In[85]:


error_rate = []


# In[86]:


for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(xtrain, ytrain)
    pred_i = knn.predict(xtest)
    error_rate.append(np.mean(pred_i != ytest))


# In[88]:


plt.figure(figsize=(10,8))
plt.plot(range(1,40), error_rate, color='black', linestyle='dashed', marker='o', markerfacecolor='pink', markersize = 10)
plt.title('Error Rate v/s k.Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:




