#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr


# In[2]:


df=pd.read_csv('time_scores.csv')
df


# In[3]:


# Here 'Hours' is a target variable and 'Scores' is predictor variable. 
# So we will take 'Hours' on x-axis and 'Scores' on y-axis.
# Setting up the axis

plt.scatter(x="Hours",y="Scores",data=df,c="green")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Simple Linear Regression Task")
plt.show()


# In[4]:


X=df.Hours.values.reshape(25,1).tolist()
y=df.Scores.values.reshape(25,1).tolist()


# In[5]:


# Using train_test_split function from sklearn to generate training dataset and testing dataset.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[6]:


# Now apply Linear Regression model

linear_reg=lr().fit(X_train,y_train)
print("Training Successful")


# In[7]:


# Plotting the Regression Line

# y=mx+c 

est_line=linear_reg.coef_*X+linear_reg.intercept_

#test data plot

plt.scatter(X,y,c="green")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Linear Regression Plot')
plt.plot(X,est_line,c="red",linewidth=2,label="regression line")
plt.show()


# # Predictions

# In[8]:



y_pred=linear_reg.predict(X_test).flatten().tolist()
y_test=np.array(y_test).flatten().tolist()


# In[9]:


# compare Actual vs Predicted

df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[10]:


# now predict Score for 9.5 hrs of study

hrs=9.25
new_pred=linear_reg.predict([[hrs]])
print("No. of Hours={}".format(hrs))
print("Predicted Score={}".format(new_pred[0]))


# # Model Evaluation

# In[11]:


# Evaluating the performance of this model

from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




