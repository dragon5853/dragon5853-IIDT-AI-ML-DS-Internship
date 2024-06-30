#!/usr/bin/env python
# coding: utf-8

# # Data Loading and Preprocessing
# 

# Load the dataset

# import pandas as pd
# data = pd.read_csv(r"C:\Users\pavan\OneDrive\Desktop\IIDT AIML\heart-disease.csv")

# Check for missing values:

# In[5]:


missing_values = data.isnull().sum()
print(missing_values)


# Convert categorical variables into dummy variables:

# In[6]:


data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
print(data.head())


# # Data Analysis

# Calculate the average age of patients with and without heart disease:

# In[7]:


average_age_with_disease = data[data['target'] == 1]['age'].mean()
average_age_without_disease = data[data['target'] == 0]['age'].mean()
print(f"Average age with heart disease: {average_age_with_disease}")
print(f"Average age without heart disease: {average_age_without_disease}")


# Determine the distribution of chest pain types among patients:

# In[8]:


chest_pain_distribution = data[['cp_1', 'cp_2', 'cp_3']].sum()
print(chest_pain_distribution)


# Find the correlation between thalach (maximum heart rate) and age:

# In[9]:


correlation_thalach_age = data['thalach'].corr(data['age'])
print(f"Correlation between thalach and age: {correlation_thalach_age}")


# Analyze the effect of sex on the presence of heart disease:

# In[10]:


heart_disease_by_sex = data.groupby('sex')['target'].mean()
print(heart_disease_by_sex)


# In[ ]:





# # Data Visualization

# Plot a histogram of the age distribution of patients:

# In[11]:


import matplotlib.pyplot as plt

plt.hist(data['age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Patients')
plt.show()


# Create a bar chart showing the distribution of chest pain types among patients:

# In[12]:


chest_pain_distribution.plot(kind='bar')
plt.xlabel('Chest Pain Type')
plt.ylabel('Frequency')
plt.title('Distribution of Chest Pain Types')
plt.show()


# Plot a scatter plot to show the relationship between thalach (maximum heart rate) and age:

# In[13]:


plt.scatter(data['age'], data['thalach'])
plt.xlabel('Age')
plt.ylabel('Thalach (Maximum Heart Rate)')
plt.title('Thalach vs. Age')
plt.show()


# Create a box plot to compare the age distribution of patients with and without heart disease:

# In[14]:


data.boxplot(column='age', by='target')
plt.xlabel('Heart Disease')
plt.ylabel('Age')
plt.title('Age Distribution by Heart Disease')
plt.suptitle('')
plt.show()


# In[ ]:





# # Advanced Analysis (using numpy)
# 

# Calculate the correlation matrix for all numerical features in the dataset:

# In[15]:


import numpy as np

correlation_matrix = data.corr()
print(correlation_matrix)


# Perform a rolling mean analysis on the chol (cholesterol) levels with a window size of 5 and plot it:

# In[16]:


data['chol_rolling_mean'] = data['chol'].rolling(window=5).mean()
plt.plot(data['chol'], label='Cholesterol')
plt.plot(data['chol_rolling_mean'], label='Rolling Mean (window=5)', color='red')
plt.xlabel('Index')
plt.ylabel('Cholesterol')
plt.title('Rolling Mean Analysis of Cholesterol Levels')
plt.legend()
plt.show()


# In[ ]:





# # Bonus

# • Create a function that can take a patient’s data as input and return a prediction of whether they have heart disease based on   simple thresholding rules.
# 
# • Use subplots to combine multiple visualizations into one figure for better comparison.

# In[18]:


def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    if age > 50 and chol > 240:
        return 1
    else:
        return 0
    
prediction = predict_heart_disease(63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)
print(f"Heart disease prediction: {prediction}")


# In[19]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(data['age'], bins=20, edgecolor='k')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution of Patients')


chest_pain_distribution.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_xlabel('Chest Pain Type')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Chest Pain Types')


axes[1, 0].scatter(data['age'], data['thalach'])
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Thalach (Maximum Heart Rate)')
axes[1, 0].set_title('Thalach vs. Age')


data.boxplot(column='age', by='target', ax=axes[1, 1])
axes[1, 1].set_xlabel('Heart Disease')
axes[1, 1].set_ylabel('Age')
axes[1, 1].set_title('Age Distribution by Heart Disease')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.show()


# In[ ]:




