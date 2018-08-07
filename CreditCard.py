
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("D:\\01_DataScience\\project\\creditcard.csv")


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.describe()


# In[7]:


#10% of data#dont run here now i am going to take all data not 10%
seed=43
data=data.sample(frac=0.1,random_state=1)


# In[6]:


print(data.shape)


# In[7]:


data.hist(column='Class')


# In[8]:


#determine number of fraud transcation
fraud=data[data['Class']== 1]
valid=data[data['Class']==0]


#outlier=len(fraud)/float(len(valid))
#print(outlier)
print('fraud cases={}'.format(len(fraud)))
print('valid cases={}'.format(len(valid)))


# In[9]:


#corelation Matrix
cormat=data.corr()
sns.heatmap(cormat,xticklabels=cormat.columns.values,yticklabels=cormat.columns.values )
plt.figure(figsize=(30,10))
plt.show()
            
        


# In[11]:


data['Amount'].corr(data['V20'])


# In[12]:


data['V7'].corr(data['Amount'])


# In[13]:


#columns=data.columns.tolist()


# In[14]:


columns


# In[10]:


#columns=[c for c in columns if c not in ['Class']]
#target='Class'
#X=data[columns]
#Y=data[target]
#print(data.shape)
#print(X.shape)
#print(Y.shape)

#we can try like that 
array=data.values
X=array[:,0:30]
y=array[:,30]
seed=7
print(data.shape)
print(X.shape)
print(y.shape)


# In[16]:


#building model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[11]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[11]:


#len(data['Class']==1)


# In[13]:


#len(data['Class']== 0)


# In[12]:


data.isnull().values.any()


# In[13]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[14]:


y_pred = logreg.predict(X_test)


# In[15]:


logreg.score(X_test, y_test)


# In[16]:


#cross validation to Logistics Regression to avoid overfitting(K -hold technique)

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.6f" % (results.mean()))


# In[17]:


#building model
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[18]:


#confusion matrix shows 85258+77 are corrected predication while 70+11 are wrong predication
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[19]:


#logit regressiom model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

