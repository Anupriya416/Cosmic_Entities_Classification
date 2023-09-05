#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('star_classification.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report,ConfusionMatrixDisplay


# In[9]:


df=pd.read_csv(r"C:\Users\hverm\Downloads\star_classification.csv")
df


# In[11]:


# DATA PREPROCESSING


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# In[14]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['class'] = LE.fit_transform(df['class'])


# In[15]:


df['class'].value_counts()


# In[16]:


# Data Visualization


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1,figsize=(20,12))
sns.heatmap(df.corr(),annot=True,linewidths=1)
plt.show()


# In[18]:


sns.countplot(x='class',data=df)


# In[19]:


import seaborn as sns

sns.distplot(df['delta'], color="midnightblue")
plt.show()


# In[20]:


#  FINDING VALUES HAVING LEAST CORRELATION


# In[21]:


#Calculate the correlation matrix
correlation_matrix = df.corr()

# Find features with high correlation
threshold = 0.5  # Adjust the threshold as needed
low_correlated_features = np.where(np.abs(correlation_matrix) > threshold)

# Print the high correlated features
for feature1, feature2 in zip(low_correlated_features[0], low_correlated_features[1]):
    if feature1 != feature2:
        print(f"{df.columns[feature1]} and {df.columns[feature2]} are low correlated.")


# In[22]:


# FEATURE SELECTION BASED ON ABOVE VARIABLES


# In[23]:


X = df[['u', 'g', 'r', 'i', 'z', 'redshift', 'plate']]
X


# In[24]:


y = df['class']
y


# In[25]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 30, k_neighbors = 5)
X_res, y_res = sm.fit_resample(X, y)


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state = 30)


# In[27]:


# MODEL EVALUVATION
# K NEAREST NEIGHBORS


# In[28]:


knn1=KNeighborsClassifier(algorithm='auto',n_neighbors=9,weights='distance')
knn1.fit(X_train,y_train)
y_pred1=knn1.predict(X_test)
print(classification_report(y_test,y_pred1))


# In[29]:


nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred2=nb.predict(X_test)
y_pred2


# In[30]:


#  GAUSSIAN NAIVE BASE


# In[31]:


print(classification_report(y_test,y_pred2))


# In[32]:


# DECISION TREE CLASSIFIER


# In[33]:


dt=DecisionTreeClassifier(criterion='entropy',random_state=2,max_depth=10)
dt.fit(X_train,y_train)
y_pred3=dt.predict(X_test)
y_pred3


# In[34]:


print(classification_report(y_test,y_pred3))


# In[35]:


# RANDOM FOREST


# In[36]:


rf=RandomForestClassifier(criterion= 'entropy', max_depth= None, min_samples_leaf= 1, min_samples_split= 4,n_estimators= 200)
rf.fit(X_train,y_train)
y_pred4=rf.predict(X_test)
y_pred4


# In[37]:


print(classification_report(y_test,y_pred4))


# In[38]:


# XG BOOST


# In[39]:


xgb=XGBClassifier()
xgb.fit(X_train,y_train)
y_pred7=xgb.predict(X_test)
y_pred7


# In[40]:


print(classification_report(y_test,y_pred7))


# In[42]:


# THE HIGEST ACCURACY IS IN KNN,XGBOOST,DECISION TREE CLASSIFIER,DECISION TREE CLASSIFIER WITH 98 %
# THANKU.....

