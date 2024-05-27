#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[7]:


data = pd.read_excel("C:\\Users\\aravind murali\\Downloads\\Telegram Desktop\\Q1.xlsx")


# In[8]:


df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=['sales', 'salary'], drop_first=True)

X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

print("Random Forest Classifier:")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\nGradient Boosting Classifier:")
print(classification_report(y_test, gb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_preds))

def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()

plot_feature_importance(rf_model, X.columns)
plot_feature_importance(gb_model, X.columns)


# In[ ]:





# In[ ]:


#Question 2


# In[9]:


data = pd.read_excel("C:\\Users\\aravind murali\\Downloads\\Telegram Desktop\\Q2.xlsx")
df = pd.DataFrame(data)

print(df.isnull().sum())

X = df.drop(['name', 'status'], axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


plt.figure(figsize=(6, 4))
sns.countplot(x='status', data=df)
plt.title('Distribution of Target Variable (Status)')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(X.columns):
    plt.subplot(4, 6, i+1)
    sns.boxplot(x='status', y=column, data=df)
    plt.title(column)
plt.tight_layout()
plt.show()


# In[13]:


missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)


df.fillna(df.mean(), inplace=True)


# In[16]:


from scipy.stats import zscore
z_scores = zscore(df.select_dtypes(include=np.number))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]


# In[19]:


rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

print("Random Forest Classifier:")
print(classification_report(y_test, rf_pred))

print("Gradient Boosting Classifier:")
print(classification_report(y_test, gb_pred))


# In[ ]:





# In[23]:


#Question 3

url = "C:\\Users\\aravind murali\\Downloads\\Telegram Desktop\\Shopping_Revenue.csv"
df = pd.read_csv(url)

print(df.head())


# In[24]:


plt.figure(figsize=(10, 6))
sns.histplot(df['revenue'], bins=20, kde=True)
plt.title('Distribution of Revenue')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


#Question 4


# In[29]:


google_sheet_link = pd.read_excel("C:\\Users\\aravind murali\\Downloads\\Telegram Desktop\\Batchwise Attendance Data.xlsx")


df = pd.DataFrame(google_sheet_link)

df.fillna('Missed', inplace=True)

df['Type'].replace('', 'STUDENT', inplace=True)

latest_columns = ['Student Roll Num', 'Type']
date_columns = [col for col in df.columns if col not in latest_columns]
df = df[latest_columns + date_columns[::-1]]

print(df.head())


# In[31]:


attendance_data = df[date_columns].apply(lambda x: x.value_counts()).T
attendance_data.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Overall Attendance Summary')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Attendance')
plt.tight_layout()
plt.show()


# In[35]:


missing_attendance = df[df[date_columns] == 'Missed'][['Student Roll Num', 'Type']].dropna()
students_missing = missing_attendance.groupby(['Student Roll Num', 'Type']).size().unstack().fillna(0)
students_missing['Total Missed'] = students_missing.sum(axis=1)
students_missing.sort_values(by='Total Missed', ascending=False, inplace=True)


# In[42]:


missing_attendance = df[df[date_columns] == 'Missed'][['Student Roll Num', 'Type']]

students_missing = missing_attendance.groupby('Student Roll Num').size().sort_values(ascending=False)

if not students_missing.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=students_missing.index, y=students_missing, palette='Reds_r')
    plt.title('Students Needing Attention for Missing Sessions')
    plt.xlabel('Student Roll Num')
    plt.ylabel('Total Missed Sessions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No missing attendance data.")


# In[44]:


satisfaction_columns = df.filter(like='R-').columns
satisfaction_data = df[satisfaction_columns]
satisfaction_counts = satisfaction_data.apply(pd.value_counts).fillna(0).T
satisfaction_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Satisfaction Trends')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[45]:


unsatisfied_students = df[df.filter(like='R-').lt(5).any(axis=1)]
very_satisfied_students = df[df.filter(like='R-').ge(8).all(axis=1)]

print("Unsatisfied Students:")
print(unsatisfied_students)

print("\nVery Satisfied Students:")
print(very_satisfied_students)


# In[ ]:




