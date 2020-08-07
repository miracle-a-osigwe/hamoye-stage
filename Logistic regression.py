#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#using python to download the dataset (uncomment the codes below and execute them)
#import urllib.request
#url = 'https://query.data.world/s/wh6j7rxy2hvrn4ml75ci62apk5hgae'
#filename = 'dataworld.csv'
#urllib.request.urlretrieve(url, filename)


# In[3]:


#using pandas to download the dataset (uncomment the code below and execute them)
#df = pd.read_csv('https://query.data.world/s/wh6j7rxy2hvrn4ml75ci62apk5hgae')
#df.to_csv('dataset.csv')


# In[4]:


#load the dataset
df = pd.read_csv('dataset.csv', low_memory=False)
df.head()


# In[5]:


#check distribution of target variable
df['QScore'].value_counts()


# In[6]:


#checking for null values in the dataset
df.isna().sum()


# In[7]:


#for simplicity, we will drop the rows with missing values.
df.dropna(inplace = True)
df.isna().sum()
#An obvious change in our target variable after removing the missing values is that there are only three classes left 
#and from the distribution of the 3 classes, we can see that there is an obvious imbalance between the classes. 
#There are methods that can be applied to handle this imbalance such as oversampling and undersampling.
#Oversampling involves increasing the number of instances in the class with fewer instances while undersampling 
#involves reducing the data points in the class with more instances.


# In[8]:


#drop the unnamed column
#reset the dataframe index
df.drop('Unnamed: 0', axis = 1, inplace = True)
df = df.reset_index(drop = True)


# In[9]:


#For now, we will convert this to a binary classification problem by combining class '2A' and '1A'.
df['QScore'] = df.loc[:, 'QScore'].replace(['1A'], '2A')
df.QScore.value_counts()


# In[10]:


df_2A = df[df.QScore=='2A']
df_3A = df[df.QScore=='3A'].sample(350)
data_df = df_2A.append(df_3A)
data_df


# In[11]:


import sklearn.utils
data_df = sklearn.utils.shuffle(data_df)
data_df = data_df.reset_index(drop=True)
data_df.shape
data_df.QScore.value_counts()


# In[12]:


#more preprocessing
data_df = data_df.drop(columns=['country_code', 'country', 'year'])
X = data_df.drop(columns='QScore')
y = data_df['QScore']


# In[13]:


#split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train.value_counts()


# In[14]:


#installing imblearn module (uncomment the pip command to install imblearn)
#!pip install imblearn


# In[15]:


#the record feature needs to be encoded
x_train.loc[:, 'record']


# In[16]:


#encode categorical variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train.record = encoder.fit_transform(x_train.record)
x_test.record = encoder.transform(x_test.record)


# In[17]:


#the encoded feature
x_train.record


# In[18]:


#There is still an imbalance in the class distribution. For this, we use SMOTE only on the training data to handle this.
import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=1)
x_train_balanced, y_balanced = smote.fit_sample(x_train, y_train)
y_train.value_counts()


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_train_df = scaler.fit_transform(x_train_balanced.drop(columns=['record']))
normalised_train_df = pd.DataFrame(normalised_train_df, columns=x_train_balanced.drop(columns=['record']).columns)
normalised_train_df['record'] = x_train_balanced['record']


# In[20]:


x_test = x_test.reset_index(drop=True)
normalised_test_df = scaler.transform(x_test.drop(columns=['record']))
normalised_test_df = pd.DataFrame(normalised_test_df, columns=x_test.drop(columns=['record']).columns)
normalised_test_df['record'] = x_test['record']


# In[21]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(normalised_train_df, y_balanced)
#returns
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


# In[ ]:




