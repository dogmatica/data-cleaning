#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn
from sklearn.decomposition import PCA


# In[6]:


df = pd.read_csv('churn_raw_data.csv',dtype={'locationid':np.int64})


# In[7]:


df.info()


# In[8]:


print (df)


# In[9]:


df.duplicated().sum()


# In[10]:


df.isnull().sum()


# In[11]:


plt.hist(df['Children'])


# In[12]:


plt.hist(df['Age'])


# In[13]:


plt.hist(df['Income'])


# In[14]:


plt.hist(df['Tenure'])


# In[15]:


plt.hist(df['Bandwidth_GB_Year'])


# In[16]:


df['Age'].fillna(df['Age'].mean(), inplace = True)
df['Children'].fillna(df['Children'].median(), inplace = True)
df['Income'].fillna(df['Income'].median(), inplace = True)
df['Tenure'].fillna(df['Tenure'].mean(), inplace = True)
df['Bandwidth_GB_Year'].fillna(df['Bandwidth_GB_Year'].mean(), inplace = True)


# In[17]:


df['Techie'] = df['Techie'].fillna(df['Techie'].mode()[0])
df['Phone'] = df['Phone'].fillna(df['Phone'].mode()[0])
df['TechSupport'] = df['TechSupport'].fillna(df['TechSupport'].mode()[0])


# In[18]:


df['Children_z'] = stats.zscore(df['Children'])
df['Age_z'] = stats.zscore(df['Age'])
df['Income_z'] = stats.zscore(df['Income'])
df['Outage_sec_perweek_z'] = stats.zscore(df['Outage_sec_perweek'])
df['Email_z'] = stats.zscore(df['Email'])
df['Contacts_z'] = stats.zscore(df['Contacts'])
df['Yearly_equip_failure_z'] = stats.zscore(df['Yearly_equip_failure'])
df['Tenure_z'] = stats.zscore(df['Tenure'])
df['MonthlyCharge_z'] = stats.zscore(df['MonthlyCharge'])
df['Bandwidth_GB_Year_z'] = stats.zscore(df['Bandwidth_GB_Year'])


# In[19]:


len(df.query('Children_z > 3 | Children_z < -3'))


# In[20]:


len(df.query('Age_z > 3 | Age_z < -3'))


# In[21]:


len(df.query('Income_z > 3 | Income_z < -3'))


# In[22]:


len(df.query('Outage_sec_perweek_z > 3 | Outage_sec_perweek_z < -3'))


# In[23]:


len(df.query('Email_z > 3 | Email_z < -3'))


# In[24]:


len(df.query('Contacts_z > 3 | Contacts_z < -3'))


# In[25]:


len(df.query('Yearly_equip_failure_z > 3 | Yearly_equip_failure_z < -3'))


# In[26]:


len(df.query('Tenure_z > 3 | Tenure_z < -3'))


# In[27]:


len(df.query('MonthlyCharge_z > 3 | MonthlyCharge_z < -3'))


# In[28]:


len(df.query('Bandwidth_GB_Year_z > 3 | Bandwidth_GB_Year_z < -3'))


# In[29]:


plt.hist(df['Children_z'])


# In[30]:


plt.hist(df['Age_z'])


# In[31]:


plt.hist(df['Income_z'])


# In[32]:


plt.hist(df['Outage_sec_perweek_z'])


# In[33]:


plt.hist(df['Email_z'])


# In[34]:


plt.hist(df['Contacts_z'])


# In[35]:


plt.hist(df['Yearly_equip_failure_z'])


# In[36]:


plt.hist(df['Tenure_z'])


# In[37]:


plt.hist(df['MonthlyCharge_z'])


# In[38]:


plt.hist(df['Bandwidth_GB_Year_z'])


# In[39]:


boxplot=seaborn.boxplot(x='MonthlyCharge',data=df)


# In[40]:


boxplot=seaborn.boxplot(x='Yearly_equip_failure',data=df)


# In[41]:


boxplot=seaborn.boxplot(x='Email',data=df)


# In[42]:


boxplot=seaborn.boxplot(x='Income',data=df)


# In[43]:


boxplot=seaborn.boxplot(x='Children',data=df)


# In[44]:


boxplot=seaborn.boxplot(x='Contacts',data=df)


# In[45]:


boxplot=seaborn.boxplot(x='Outage_sec_perweek',data=df)


# In[46]:


churn_MonthlyCharge_z = df.query('MonthlyCharge_z > 3 | MonthlyCharge_z < -3')
churn_MonthlyCharge_z_sort = churn_MonthlyCharge_z.sort_values(['MonthlyCharge'], ascending = False)
churn_MonthlyCharge_z_sort.to_csv(r'C:\Users\wstul\d206\churn_MonthlyCharge_z_sort.csv')


# In[47]:


churn_Yearly_equip_failure_z = df.query('Yearly_equip_failure_z > 3 | Yearly_equip_failure_z < -3')
churn_Yearly_equip_failure_z_sort = churn_Yearly_equip_failure_z.sort_values(['Yearly_equip_failure'], ascending = False)
churn_Yearly_equip_failure_z_sort.to_csv(r'C:\Users\wstul\d206\churn_Yearly_equip_failure_z_sort.csv')


# In[48]:


churn_Email_z = df.query('Email_z > 3 | Email_z < -3')
churn_Email_z_sort = churn_Email_z.sort_values(['Email'], ascending = False)
churn_Email_z_sort.to_csv(r'C:\Users\wstul\d206\churn_Email_z_sort.csv')


# In[49]:


churn_Income_z = df.query('Income_z > 3 | Income_z < -3')
churn_Income_z_sort = churn_Income_z.sort_values(['Income'], ascending = False)
churn_Income_z_sort.to_csv(r'C:\Users\wstul\d206\churn_Income_z_sort.csv')


# In[50]:


churn_Children_z = df.query('Children_z > 3 | Children_z < -3')
churn_Children_z_sort = churn_Children_z.sort_values(['Children'], ascending = False)
churn_Children_z_sort.to_csv(r'C:\Users\wstul\d206\churn_Children_z_sort.csv')


# In[51]:


churn_Contacts_z = df.query('Contacts_z > 3 | Contacts_z < -3')
churn_Contacts_z_sort = churn_Contacts_z.sort_values(['Contacts'], ascending = False)
churn_Contacts_z_sort.to_csv(r'C:\Users\wstul\d206\churn_Contacts_z_sort.csv')


# In[52]:


churn_outage_sec_z = df.query('Outage_sec_perweek_z > 3 | Outage_sec_perweek_z < -3')
churn_outage_sec_z_sort = churn_outage_sec_z.sort_values(['Outage_sec_perweek_z'], ascending = False)
churn_outage_sec_z_sort.to_csv(r'C:\Users\wstul\d206\churn_outage_sec_z_sort.csv')


# In[53]:


df.drop(['Children_z', 'Age_z', 'Income_z', 'Outage_sec_perweek_z', 'Email_z', 'Contacts_z', 'Yearly_equip_failure_z', 'Tenure_z', 'MonthlyCharge_z', 'Bandwidth_GB_Year_z'], axis=1, inplace=True)


# In[54]:


df_marital_ohe = pd.get_dummies(df['Marital'], prefix = 'Marital', drop_first = False)
df_marital_ohe


# In[55]:


df = pd.concat([df, df_marital_ohe], axis = 1)
df


# In[56]:


df.drop(['Marital'], axis=1, inplace=True)


# In[57]:


Education
scale_mapper = {'Month-to-month' : 1, 'One year' : 2, 'Two Year' : 3}
df['Contract_Duration'] = df['Contract'].replace(scale_mapper)
df['Contract_Duration']


# In[58]:


scale_mapper = {'Month-to-month' : 1, 'One year' : 2, 'Two Year' : 3}
df['Contract_Duration'] = df['Contract'].replace(scale_mapper)
df['Contract_Duration']


# In[59]:


scale_mapper = {'No Schooling Completed' : 1, 'Nursery School to 8th Grade' : 2, '9th Grade to 12th Grade, No Diploma' : 3, 'GED or Alternative Credential' : 4, 'Regular High School Diploma' : 5, 'Some College, Less than 1 Year' : 6, 'Some College, 1 or More Years, No Degree' : 7, 'Professional School Degree' : 8, "Associate's Degree" : 9, "Bachelor's Degree" : 10, "Master's Degree" : 11, 'Doctorate Degree' : 12}
df['Education_Level'] = df['Education'].replace(scale_mapper)
df['Education_Level']


# In[60]:


test_pca = df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Education_Level', 'Contract_Duration']]


# In[61]:


test_pca_normalized = (test_pca - test_pca.mean()) / test_pca.std()


# In[62]:


pca = PCA(n_components = test_pca.shape[1])
pca.fit(test_pca_normalized)


# In[63]:


test_pca2 = pd.DataFrame(pca.transform(test_pca_normalized), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])


# In[64]:


loadings = pd.DataFrame(pca.components_.T, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'], index = test_pca_normalized.columns)
loadings


# In[65]:


cov_matrix = np.dot(test_pca_normalized.T, test_pca_normalized) / test_pca.shape[0]


# In[66]:


eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]


# In[67]:


plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalues')
plt.show()


# In[68]:


df.to_csv(r'C:\Users\wstul\d206\William_Stults_churn_raw_data_cleaned.csv')


# In[ ]:




