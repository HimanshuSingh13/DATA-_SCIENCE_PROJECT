#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION TO DATA SET
# 
# The Iris Dataset contains four features (length and width of sepals and petals) of 150 samples of three species of Iris ie classes (Iris setosa, Iris virginica and Iris versicolor).
# 
# ![image.png](attachment:image.png)

# To featch data from the given data set file.

# In[1]:


import numpy as np
import pandas as pd


# ## To load the data

# In[20]:


data=pd.read_csv("D:\\gitsessions\\Iris.CSV")
data


# In[6]:


data.head()  # to get the top 5 rows


# ##   01) To Segregate features and classes

# In[13]:


features=list(data.columns)
features


# In[19]:


classes=list(data.Species.unique())
classes


# ## 02) Spliting the data into train and test data in the ratio 70:30

# we have to shuffle the dataset to assure an even distribution of classes when splitting the dataset into training and test set. 

# In[21]:


data = data.sample(frac=1, random_state=42)
data.set_index("Id", inplace=True)
data.head()


# In[26]:


train_dataset=data[:106]
test_dataset=data[106:]


# In[31]:


train_dataset.info()


# In[33]:


train_dataset.describe()


# In[34]:


test_dataset.info()


# In[35]:


test_dataset.describe()


# ## 03) Train the "RandomForestClassifier" and "LogisticRegression"

# ### DATA PREPRATION
# 
# #### First we have to save our features and our target variable in separate dataframes for both the training and the test set.

# In[40]:


X_train = train_dataset.drop("Species", axis=1)
y_train = train_dataset["Species"].copy()

X_test = test_dataset.drop("Species", axis=1)
y_test = test_dataset["Species"].copy()


# In[44]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("impute", SimpleImputer(strategy="mean"))
])


# In[45]:


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("full", my_pipeline, X_train.columns)
])


# In[46]:


X_train_prepared = full_pipeline.fit_transform(X_train)


# ## "RANDOM FOREST CLASSIFIER"

# In[59]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

forest_clf = RandomForestClassifier()

scores_forest = cross_val_score(forest_clf, X_train_prepared, y_train, cv=4, scoring="accuracy")


# In[60]:


print(scores_forest)
print("mean: ", scores_forest.mean())
print("Std: ", scores_forest.std())


# ## "LOGISTIC REGRESSION"

# In[61]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

scores_log = cross_val_score(log_reg, X_train_prepared, y_train, cv=4, scoring="accuracy")
print(scores_log)
print("mean: ", scores_log.mean())
print("Std: ", scores_log.std())


# ## 04)  CLEARLY FROM THE ABOVE RESULT "RANDOM FOREST CLASSIFIER" HAVE MORE ACCURACY THAN "LOGISTIC REGRESSION"
# 
# ### RANDOM FOREST CLASSIFIER
# [1.         0.96296296 0.96153846 0.88461538]
# mean:  0.9522792022792023
# Std:  0.04199864099089907
# 
# ### LOGISTIC REGRESSION
# [1.         0.96296296 0.96153846 0.84615385]
# mean:  0.9426638176638177
# Std:  0.05781418486628931

# In[ ]:





# In[ ]:




