#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


na_vals = ['NA','Missing']

df = pd.read_csv('dev_survey_data/survey_results_public.csv',index_col = 'Respondent', na_values = na_vals)


# In[3]:


# Dataframe are rows and columns of data


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


pd.set_option('display.max_columns',85)
pd.set_option('display.max_rows',85)


# In[8]:


df


# In[9]:


schema_df = pd.read_csv('dev_survey_data/survey_results_schema.csv')


# In[10]:


schema_df


# In[11]:


df


# In[12]:


df


# In[13]:


df.head()


# In[14]:


df.tail()


# In[15]:


pd.set_option('display.max_rows',85)


# In[16]:


df


# In[17]:


df1 = pd.read_csv('dev_survey_data/survey_results_public.csv')


# In[18]:


df1


# In[19]:


df


# In[20]:


# DataFrame => rows and columns of data.(2D data structure)


# In[21]:


people = {
    'first' : ['Corey', 'Jane', 'John'],
    'last' : ['Schafer', 'Doe', 'Doe'],
    'email' : ['CoreySchafer@gmail.com','JaneDoe@gmail.com','JohnDoe@gmail.com']
}


# In[22]:


people['email']


# In[23]:


dframe = pd.DataFrame(people)


# In[24]:


dframe


# In[25]:


dframe['email']
# dframe['email'][1]


# In[26]:


# The DataFrame is very similar, equivalent to the dictionary. The dictionary is returning list, but dataframe is returning 
# "Series" of values
type(dframe['email'])


# In[27]:


# As a dataframe is similar to dict, a series is a list of data, with more functionality
# Series is a 1-D Array


# In[28]:


# DataFrame => Rows, and Columns; Series=> Rows of a single column
# DataFrame is a container of multiple of these series object.


# In[29]:


dframe[['last','email']]


# In[30]:


dframe.columns


# In[31]:


# dframe.iloc
#access rows by integer location
dframe.iloc[0]


# In[32]:


dframe.iloc[[0,1]]


# In[33]:


dframe.iloc[[0,1],2]


# In[34]:


# with loc we can search with labels


# In[35]:


dframe.loc[[0,1]]


# In[36]:


dframe.loc[[0,1],'email']


# In[37]:


dframe.loc[[0,1],['first','email']]


# In[38]:


df


# In[39]:


df['Hobbyist']


# In[40]:


# To count how many yes, or no in Hobbyist series:
df['Hobbyist'].value_counts()


# In[41]:


df.loc[[0,1,2], ['Hobbyist']]


# In[ ]:


df.loc[0:2,'Hobbyist']


# In[ ]:


df.loc[0:2,'Hobbyist':'Employment']


# In[ ]:


people


# In[ ]:


dframe


# # Index

# In[ ]:


# To set email as index:


# In[ ]:


dframe.set_index('email',inplace = True)


# In[ ]:


dframe


# In[ ]:


dframe.index


# In[ ]:


dframe.loc['CoreySchafer@gmail.com']


# In[ ]:


dframe.reset_index(inplace=True)


# In[ ]:


dframe


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


df.set_index('Respondent',inplace = True)


# In[ ]:


df


# In[ ]:





# In[ ]:


schema_df.set_index('Column',inplace = True)


# In[ ]:


schema_df


# In[ ]:


schema_df.loc['Hobbyist']


# In[ ]:


schema_df.loc['Respondent']


# In[ ]:


schema_df.loc['MgrIdiot','QuestionText']


# In[ ]:


schema_df.sort_index(inplace = True)


# In[ ]:


schema_df


# In[ ]:


schema_df.loc['BlockchainIs','QuestionText']


# In[ ]:


schema_df['QuestionText']


# # FILTER

# In[ ]:


dframe


# In[ ]:


dframe['last'] == 'Doe'


# In[ ]:


filt = (dframe['last'] == 'Doe') 
dframe[filt]


# In[ ]:


dframe.loc[filt,'email']


# In[ ]:


dframe.loc[(dframe['last'] == 'Doe') & (dframe['first'] == 'John')]


# In[ ]:


dframe.loc[(dframe['last']=='Schafer') | (dframe['first'] == 'John'), 'email']


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


high_salary = (df['ConvertedComp'] > 70000)


# In[ ]:


df.loc[high_salary,['Country','LanguageWorkedWith','ConvertedComp']]


# In[ ]:


countries = ['United States','India','	United Kingdom','Germany','Canada']
filt2 = df['Country'].isin(countries)


# In[ ]:


df.loc[filt2,'Country']


# In[ ]:


df['LanguageWorkedWith']


# In[ ]:


filt3 = df['LanguageWorkedWith'].str.contains('Python',na=False)


# In[ ]:


df.loc[filt3,'LanguageWorkedWith']


# In[ ]:


df.loc[8,'LanguageWorkedWith']


# # UPDATING ROWS AND COLUMNS

# In[ ]:


dframe


# In[ ]:


dframe.columns


# In[ ]:


dframe.columns = ['email', 'firstname', 'lastname']


# In[ ]:


dframe


# In[ ]:


dframe.columns = [x.lower() for x in dframe.columns]


# In[ ]:


dframe


# In[ ]:


# dframe.columns = df.columns.str.replace(' ','_')


# In[ ]:


dframe.rename(columns = {'firstname':'first','lastname':'last'},inplace = True)


# In[ ]:


dframe


# In[ ]:


dframe.loc[2,'last'] = 'Smith'


# dframe

# In[ ]:


dframe.loc[2,['last','email']] = ['Smith', 'JohnSmith@gmail.com']


# In[ ]:


dframe


# In[ ]:


dframe['email'].str.lower()


# In[ ]:


dframe['email'] = dframe['email'].str.lower()


# In[ ]:


dframe


# ###### FOUR METHODS
# 1.apply
# 2.map
# 3.applymap
# 4.replace

# # APPLY

# In[ ]:


# for series: Apply a function for every value of a series


# In[ ]:


dframe['email'].apply(len)


# In[ ]:


def update_email(email):
    return email.upper()


# In[ ]:


dframe['email'].apply(update_email)


# In[ ]:


dframe['email'] = dframe['email'].apply(update_email)


# In[ ]:


dframe


# In[ ]:


dframe['email'].apply(lambda x : x.lower())


# In[ ]:


dframe['email'] = dframe['email'].apply(lambda x : x.lower())


# In[ ]:


dframe


# In[ ]:


# for dataframes: runs a function on each row or column (or every series) of that dataframe


# In[ ]:


dframe.apply(len)


# In[ ]:


# The above function was applied on each column: It tells us how many values in each column.


# In[ ]:


dframe.apply(len, axis='columns')


# # APPLYMAP

# In[ ]:


# Only works on a Dataframe. Applies a function to every value of a dataframe


# In[ ]:


dframe.applymap(len)


# In[ ]:


dframe.applymap(str.lower)


# # MaP

# In[ ]:


# Only works on a series:


# In[ ]:


dframe['first'].map({'Corey':'Chris','Jane':'Mary'})


# In[ ]:


# So, using a map we get an Nan for values which we don't have to substitute. We can use replace instead


# # Replace

# In[ ]:


dframe['first'].replace({'Corey':'Chris','Jane':'Mary'})


# In[ ]:


dframe['first'] = dframe['first'].replace({'Corey':'Chris','Jane':'Mary'})


# In[ ]:


dframe


# In[ ]:


dframe


# In[ ]:


dframe


# In[ ]:


df


# In[ ]:


df.rename(columns={'ConvertedComp':'SalaryUSD'},inplace=True)


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


df['Hobbyist']


# In[ ]:


df['Hobbyist']=df['Hobbyist'].map({'Yes':True,'No':False})


# In[ ]:


df


# # Adding rows and columns

# In[ ]:


dframe


# In[ ]:


dframe['first'] + ' ' +dframe['last']


# In[ ]:


dframe['fullname'] = dframe['first'] + ' ' +dframe['last']


# In[ ]:


dframe


# In[ ]:


dframe.drop(columns=['first','last'], inplace=True)


# In[ ]:


dframe


# In[ ]:


dframe['fna'] = ['Cs','Md','Js']


# In[ ]:


dframe


# In[ ]:


dframe.drop(columns=['fna'], inplace =True)


# In[ ]:


dframe


# In[ ]:


dframe['fullname'].str.split(' ',expand=True)


# In[ ]:


dframe[['first','last']] = dframe['fullname'].str.split(' ',expand=True)


# In[ ]:


dframe


# In[ ]:


# To add rows:


# In[ ]:


dframe.append({'first':'Tony'},ignore_index=True)


# In[ ]:


people = {
    'first' : ['Tony', 'Steve'],
    'last' : ['Stark', 'Rogers'],
    'email' : ['IronMan@avengers.com','Cap@avengers.com']
}


# In[ ]:


df2 = pd.DataFrame(people)


# In[ ]:


df2


# In[ ]:


dframe.append(df2,ignore_index=True)


# In[ ]:


dframe = dframe.append(df2,ignore_index=True)


# In[ ]:


dframe.drop(index=4)


# In[ ]:


dframe


# In[ ]:





# In[ ]:


dframe['last'] = ['Schafer','Doe','Doe','Stark','Rogers']


# In[ ]:


dframe.drop(index=dframe[dframe['last'] == 'Doe'].index)


# In[ ]:


people = {
    'firstn' : ['Tony', 'Steve'],
    'lastn' : ['Stark', 'Rogers'],
    'emailadd' : ['IronMan@avengers.com','Cap@avengers.com']
}


# In[ ]:


df4=pd.DataFrame(people)


# In[ ]:


dframe


# In[ ]:


dframe.drop(index=[3,4],inplace=True)


# In[ ]:


dframe


# #### 

# In[ ]:


df4


# In[ ]:


dframe.append(df4)


# In[ ]:


dframe


# In[ ]:


df4


# In[ ]:


df4.columns = ['first','last','email']


# In[ ]:


df4


# In[ ]:


dframe.append(df4)


# In[ ]:


dframe = dframe.append(df4,ignore_index=True)


# In[ ]:


dframe


# In[ ]:


dframe.drop(index=(dframe[dframe['last']=='Doe']).index, inplace=True)


# In[ ]:


dframe


# In[ ]:


people = {
    'first' : ['Corey', 'Jane', 'John'],
    'last' : ['Schafer', 'Doe', 'Doe'],
    'email' : ['CoreySchafer@gmail.com','JaneDoe@gmail.com','JohnDoe@gmail.com']
}


# In[ ]:


dframe = pd.DataFrame(people)


# In[ ]:


dframe


# # SORTING DATA

# In[ ]:


dframe.sort_values(by='last',ascending=True)


# In[ ]:


# We wont to sort the data by last names, and if there are same last names then we want to sort by first names:


# In[ ]:


dframe.sort_values(by=['last','first'],ascending = False)


# In[ ]:


dframe.sort_values(by=['last','first'],ascending = [False,True])


# In[ ]:


df


# In[ ]:


df.sort_values(by=['Country','SalaryUSD'],ascending=[True,False],inplace=True)


# In[ ]:


df[['Country','SalaryUSD']].head(50)


# In[ ]:


df['SalaryUSD'].nlargest(10)


# In[ ]:


df.nlargest(10,'SalaryUSD')[['WebFrameWorkedWith','SalaryUSD']]


# # GROUPING AND AGGREGATING DATA

# In[ ]:


df.sort_index(inplace=True)


# In[ ]:


df['SalaryUSD'].head(15)


# In[ ]:


# Ignores the NaN values
df['SalaryUSD'].median()


# In[ ]:


# Did not use mean because mean is affected by outliers, while median is not
df.median()


# In[ ]:


df.describe()


# In[ ]:


df.describe()['SalaryUSD']


# In[ ]:


df['Hobbyist']


# In[ ]:


df['Hobbyist'].replace({True:'Yes', False:'No'},inplace = True)


# In[ ]:


df['Hobbyist'].value_counts()


# In[ ]:


df['SocialMedia']


# In[ ]:


df['SocialMedia'].value_counts()


# In[ ]:


df['SocialMedia'].value_counts(normalize=True)


# # GROUPING

# In[ ]:


# Groupby => Involves some combination of spliting the object, applying a function, and combining the results.


# In[ ]:


df['Country'].value_counts()


# # 1.Split the objects

# In[ ]:


country_grp = df.groupby(['Country'])


# In[ ]:


df.groupby(['Country'])


# In[ ]:


country_grp.get_group('United States')


# # 2.Apply the function

# In[ ]:


country_grp['SocialMedia'].value_counts().head(50)


# In[ ]:


country_grp['SocialMedia'].value_counts().loc['India']


# In[ ]:


country_grp['SalaryUSD'].median()


# In[ ]:


country_grp['SalaryUSD'].median().loc['Germany']


# In[ ]:


country_grp['SalaryUSD'].agg(['median','mean'])


# In[ ]:


country_grp['SalaryUSD'].agg(['median','mean']).loc['Canada']


# In[ ]:


filt = df['Country'] == 'India'
df.loc[filt]['LanguageWorkedWith'].str.contains('Python').sum()


# In[ ]:


country_grp['LanguageWorkedWith'].value_counts().loc['India']


# In[ ]:


# country_grp['LanguageWorkedWith'].str.contains('Python').sum()
# The above one cannot be used because the object is no longer a series, but a series groupby object. So, apply method is used.
country_grp['LanguageWorkedWith'].apply(lambda x : x.str.contains('Python').sum())


# In[ ]:


country_resp = df['Country'].value_counts()


# In[ ]:


country_py = country_grp['LanguageWorkedWith'].apply(lambda x : x.str.contains('Python').sum())


# In[ ]:


country_resp


# In[ ]:


country_py


# In[ ]:


python_df = pd.concat([country_resp,country_py],axis='columns')


# In[ ]:


python_df


# In[ ]:


python_df.rename(columns={'Country':'Number_of_Resp','LanguageWorkedWith':'Num_knows_python'},inplace=True)


# In[ ]:


python_df


# In[ ]:


python_df['Pct_knows_python'] = (python_df['Num_knows_python']/python_df['Number_of_Resp'])*100


# In[ ]:


python_df


# In[ ]:


python_df.sort_values(by='Pct_knows_python',ascending=False,inplace = True)


# In[ ]:


python_df[python_df['Number_of_Resp'] > 100]


# # CLEANING DATA

# In[ ]:


dframe


# In[70]:


people = {
    'first' : ['Corey', 'Jane', 'John','Chris',np.nan,None,'NA'],
    'last' : ['Schafer', 'Doe', 'Doe','Schafer',np.nan,np.nan,'Missing'],
    'email' : ['CoreySchafer@gmail.com','JaneDoe@gmail.com','JohnDoe@gmail.com',None,np.nan,'Anonymous@gmail.com','NA'],
    'age' : ['33','55','63','36',None,None,'Missing']
}


# In[71]:


df2 = pd.DataFrame(people)
df2


# In[ ]:


# remove the missing data


# In[72]:


df2.dropna()


# In[73]:


# Default:
# index=>drop rows, otherwise "columns"
# any=> drop even if only one value is missing
# all = > drop only if all values of the row are missing.
df2.dropna(axis='index',how='any')


# In[74]:


df2


# In[75]:


df2.dropna(axis='index',how='all')


# In[76]:


df2.dropna(axis='columns',how='all')


# In[77]:


df2.dropna(axis='columns',how='any')


# In[78]:


# remove if email is missing/ as long as email is present should not drop that row.
df2.dropna(axis='index',how='any',subset=['email'])


# In[ ]:


# drop if any of email or last name are missing
df2.dropna(axis='index',how='any',subset=['email','last'])


# In[ ]:


# either need last name or email address, may not require both:
df2.dropna(axis='index',how='all',subset=['email','last'])


# In[ ]:


df2.replace('NA',np.nan,inplace=True)


# In[ ]:


df2.replace('Missing',np.nan,inplace=True)


# In[ ]:


df2


# In[ ]:


df2.isna()


# In[ ]:


df2.fillna('MISSING')


# # CASTING DATATYPES

# In[ ]:


people = {
    'first' : ['Corey', 'Jane', 'John','Chris',np.nan,None,'NA'],
    'last' : ['Schafer', 'Doe', 'Doe','Schafer',np.nan,np.nan,'Missing'],
    'email' : ['CoreySchafer@gmail.com','JaneDoe@gmail.com','JohnDoe@gmail.com',None,np.nan,'Anonymous@gmail.com','NA'],
    'age' : ['33','55','63','36',None,None,'Missing']
}


# In[ ]:


df3=pd.DataFrame(people)


# In[ ]:


df3


# In[ ]:


df3.replace('NA',np.nan,inplace=True)


# In[ ]:


df3.replace('Missing',np.nan,inplace=True)


# In[ ]:


df3.fillna(0)


# In[ ]:


df3.dtypes


# In[ ]:


df3['age'].mean()


# In[ ]:


type(np.nan)


# In[ ]:


# If we convert the missing values to integer it will throw error, as nan is a float


# In[ ]:


# df3['age'] = df3['age'].astype(int)=>error
df3['age'] = df3['age'].astype(float)


# In[ ]:


df3


# In[ ]:


df3.dtypes


# In[ ]:


df3['age'].mean()


# In[ ]:


df


# In[ ]:


df['YearsCode'].head(10)


# In[ ]:


df['YearsCode'] = df['YearsCode'].astype(float)


# In[ ]:


df['YearsCode'].value_counts()


# In[ ]:


df['YearsCode'].unique()


# In[ ]:


df['YearsCode'].replace('Less than 1 year',0,inplace = True)


# In[ ]:


df['YearsCode'].unique()


# In[ ]:


df['YearsCode'].replace('More than 50 years',51,inplace = True)


# In[ ]:


df['YearsCode'] = df['YearsCode'].astype(float)


# In[ ]:


df['YearsCode'].mean()


# In[ ]:


df['YearsCode'].median()


# 
# # DATETIME SERIES

# In[ ]:


df1 = pd.read_csv('dev_survey_data/ETH_1h.csv')


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# In[ ]:


df1.loc[0,'Date']


# In[ ]:


df1.loc[0,'Date'].day_name()


# In[ ]:


# df1['Date'] = pd.to_datetime(df1['Date'])


# # Read/Write from different formats

# # 1.CSV

# In[ ]:


df = pd.read_csv('dev_survey_data/survey_results_public.csv',index_col = 'Respondent')


# In[ ]:


df.head()


# In[ ]:


# WRite to CSV


# In[ ]:


filt = (df['Country'] == "India")
India_df = df.loc[filt]
India_df.head()


# In[ ]:


# To export this df to a csv file:
India_df.to_csv('dev_survey_data/modified_India.csv')


# # 2.TSV

# In[ ]:


# To export this df to a tsv file:
India_df.to_csv('dev_survey_data/modified_India.tsv',sep='\t')


# # 3.EXCEL

# In[ ]:


India_df.to_excel('dev_survey_data/modified_India.xlsx')


# In[ ]:


# read excel
test = pd.read_excel('dev_survey_data/modified_India.xlsx', index_col = 'Respondent')


# In[ ]:


test.head()


# # 4.JSON

# In[ ]:


# dict Like
India_df.to_json('dev_survey_data/modified_India.json')


# In[ ]:


# List/Records like
India_df.to_json('dev_survey_data/modified_India.json',orient='records',lines=True)


# In[ ]:


# Read json file:
test = pd.read_json('dev_survey_data/modified_India.json',orient='records',lines=True)


# In[ ]:


test


# # 5.SQL

# In[ ]:


df


# In[ ]:


(df['Country'].value_counts())['India']


# In[ ]:


pd.Series([(df['Country'].value_counts())['India'],(df['Country'].value_counts())['Germany']],index=['India','Germany'])


# In[42]:


drinks = pd.read_csv('http://bit.ly/drinksbycountry')


# In[43]:


drinks


# In[44]:


# When to use groupby?
# Whenever you want to analyse a dF by some category, in this case is the continent 
# If you ask questions like "for each": ForEach continent what is the beer service


# In[45]:


drinks.groupby('continent').beer_servings.mean()


# In[46]:


drinks.groupby('continent').mean().plot(kind='bar')


# In[51]:


drinks.groupby('continent').agg(['min','max','count','mean'])


# In[53]:


drinks.groupby('continent').mean()


# In[54]:


drinks.groupby('continent').mean().plot(kind='bar')


# In[3]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[8]:


x = [1,2,3,4,5,6,7,8,8,9]
y = np.linspace(1,10,10)
y.shape


# In[9]:


plt.scatter(x,y)


# # DATETIME

# In[2]:


import pandas as pd
df = pd.read_csv('ETH_1h.csv')
df.head()


# In[3]:


df.dtypes


# In[5]:


df.Date = pd.to_datetime(df.Date,format='%Y-%m-%d %I-%p')


# In[8]:


df.loc[0,'Date'].day_name()


# In[9]:


df.head()


# In[10]:


df.Date.apply(lambda x:x.day_name())


# In[11]:


df.Date.dt.day_name()
# to access complete column and apply class functions


# In[13]:


df['Day'] = df.Date.dt.day_name()
df.head()


# In[14]:


df['Date'].min()


# In[15]:


df['Date'].max()


# In[16]:


df['Date'].max() - df['Date'].min()


# In[20]:


df[df['Date'].dt.year >= 2020].min()


# In[21]:


df[df['Date'].dt.year >= 2020]


# In[23]:


df[df['Date'].dt.year == 2019]


# In[24]:


df.set_index('Date',inplace=True)


# In[25]:


df


# In[29]:


df.loc['2019']


# In[30]:


df.loc['2020-01':'2020-02']


# In[33]:


df.loc['2020-01':'2020-02'].Close.mean()


# In[34]:


df.loc['2020-01-01'].High.max()


# In[38]:


highs = df['High'].resample('D').max()
highs['2020-01-01']


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(highs)


# In[41]:


highs.plot()


# In[42]:


df.resample('W').mean()


# In[43]:


df.resample('W').agg({'High':'max','Close':'mean','Low':'min','Volume':'sum'})


# # EDA TITANIC 

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


df = pd.read_csv('titanic.csv')
df.head()


# In[47]:


df.shape


# In[48]:


df.isnull().sum()


# In[49]:


sns.heatmap(df.isnull(),cmap='viridis')


# In[34]:


sns.set()


# In[35]:


sns.countplot(x='Survived',data=df,hue='Sex',palette='RdBu_r')
# 1=> survived
# 0=>not survived


# In[36]:


sns.countplot(x='Survived',data=df,hue='Pclass')


# In[37]:


sns.histplot(x='Age',data=df,bins=40,color='darkred')


# In[38]:


sns.countplot(x='SibSp',data=df,hue='Survived')


# In[39]:


sns.histplot(x='Fare',data=df,bins=40,color='green')


# In[40]:


sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# In[54]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        if Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[55]:


df.Age = df[['Age','Pclass']].apply(impute_age,axis=1)


# In[56]:


df.Age


# In[57]:


sns.heatmap(df.isnull(),cmap='viridis')


# In[59]:


df.drop('Cabin',axis=1,inplace=True)


# In[60]:


sns.heatmap(df.isnull(),cmap='viridis')


# In[61]:


df.head()


# In[68]:


dummies = pd.get_dummies(df[['Embarked','Sex']],drop_first=True)
train = pd.concat([df,dummies],axis=1)


# In[70]:


train.head()


# In[71]:


train.drop(columns=['PassengerId','Name','Sex','Embarked','Ticket'],axis='columns',inplace=True)


# In[72]:


train.head()


# In[73]:


X = train.drop('Survived',axis='columns')
y = train.Survived


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[78]:


y_pred = model.predict(X_test)


# In[79]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[80]:


sns.heatmap(cm,annot=True,fmt='d')


# In[82]:


model.score(X_test,y_test)


# In[83]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[13]:


import pandas as pd
df = pd.read_csv("weather_by_cities.csv")
df.day = pd.to_datetime(df.day)
df.set_index('day',inplace=True)
df


# In[16]:


g = df.groupby('city')
print(g)


# In[22]:


for city, city_df in g:
    e = city_df.groupby('event')
    for eve, eve_df in e:
        print(eve)
        print(eve_df)


# In[87]:


g.get_group('mumbai')


# In[88]:


g.temperature.max()


# In[89]:


g.max()


# In[90]:


g.mean()


# In[91]:


india_weather = pd.DataFrame({
    "city": ["mumbai","delhi","banglore"],
    "temperature": [32,45,30],
    "humidity": [80, 60, 78]
})
india_weather


# In[27]:


import numpy as np
us_weather = pd.DataFrame({
    "city": ["new york","chicago","orlando"],
    "temperature": [np.nan,np.nan,np.nan],
    "humidity": [68, 65, 75]
})
us_weather


# In[46]:


tmean = us_weather.temperature.mean()


# In[47]:





# In[44]:


if tmean:
    tmean=234
tmean


# In[94]:


df = pd.concat([india_weather,us_weather])
df


# In[95]:


df = pd.concat([india_weather,us_weather],ignore_index=True)
df


# In[97]:


df = pd.concat([india_weather,us_weather],keys=['india','us'])
df


# In[101]:


df.loc['india']


# In[52]:


temperature_df = pd.DataFrame({
    "city": ["mumbai","delhi","banglore"],
    "temperature": [32,45,30],
}, index=[0,1,2])
temperature_df


# In[53]:


windspeed_df = pd.DataFrame({
    "city": ["delhi","mumbai",],
    "windspeed": [7,12],
}, index=[1,0])
windspeed_df


# In[54]:


df = pd.concat([temperature_df,windspeed_df])
df


# In[64]:


g = df.groupby('city')


# In[65]:


for city, city_df in g:
    print(city)
#     temp = city_df.groupby('temperature')
    df['windspeed'] = city_df.groupby('temperature').transform(lambda x:x.fillna(x.mean()))
    
#         print(temp_df)
    


# In[66]:


df


# In[51]:


tmean = df.temperature.mean()
df.temperature.fillna(tmean,inplace=True)
df


# In[106]:


df = pd.concat([temperature_df,windspeed_df],axis=1)
df


# In[107]:


temperature_df


# In[108]:


s = pd.Series(["Humid","Dry","Rain"], name="event")
s


# In[109]:


pd.concat([temperature_df,s],axis=1)


# In[110]:


df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando"],
    "temperature": [21,14,35],
})
df1


# In[111]:



df2 = pd.DataFrame({
    "city": ["chicago","new york","orlando"],
    "humidity": [65,68,75],
})
df2


# In[112]:


df3 = pd.merge(df1,df2,on='city')


# In[113]:


df3


# In[114]:


df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando","baltimore"],
    "temperature": [21,14,35,32],
})
df1


# In[115]:


df2 = pd.DataFrame({
    "city": ["chicago","new york","san fransisco"],
    "humidity": [65,68,71],
})
df2


# In[116]:


df3 = pd.merge(df1,df2,on='city')


# In[117]:


df3
# this is an inner join => intersection


# In[118]:


df3 = pd.merge(df1,df2,on='city',how='outer')
# this is an outer join => union
df3


# In[119]:



df3 = pd.merge(df1,df2,on='city',how='left')
# take common elements of left df=> df1, and common


# In[120]:


df3


# In[121]:


df3 = pd.merge(df1,df2,on='city',how='right')
df3


# In[122]:


df3 = pd.merge(df1,df2,on='city',how='outer',indicator=True)


# In[123]:


df3


# In[124]:


df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando", "baltimore"],
    "temperature": [21,14,35,38],
    "humidity": [65,68,71, 75]
})
df1


# In[125]:


df2 = pd.DataFrame({
    "city": ["chicago","new york","san diego"],
    "temperature": [21,14,35],
    "humidity": [65,68,71]
})
df2


# In[126]:


df3 = pd.merge(df1,df2,on='city')


# In[127]:


df3


# In[128]:


df3 = pd.merge(df1,df2,on='city',suffixes=['_left','_right'])


# In[130]:


df3


# In[131]:


# 1)append =>df1.append(df2) => stacking vertically
# 2)concat =>([df1,df2]) => hori or vertically, inner/outer joins on indice
# 3)join => df1.join(df2) => inner/outer, left/right joins on indices
# 4)merge => pd.merge(df1,df2) =>joins on multiple columns
# use concat instead of append, and use merge instead of join


# In[2]:


import pandas as pd
stocks = pd.read_csv('http://bit.ly/smallstocks')
stocks.head()


# In[3]:


stocks.groupby('Symbol').Close.mean()


# In[4]:


stocks.groupby(['Symbol','Date']).Close.mean()


# In[12]:


stocks.dtypes


# In[5]:


df = stocks.groupby(['Symbol','Date'])


# In[10]:


for (sym,date), sym_df in df:
    print(sym)
    print(date)
    print(sym_df)


# In[57]:


ser1 = stocks.groupby(['Symbol','Date']).agg({'Close':'mean','Volume':'mean'})
ser1.unstack()


# In[14]:


ser = stocks.groupby(['Symbol','Date']).agg({'Close':'mean','Volume':'mean'}).mean()
ser


# In[15]:


ser.unstack()


# In[16]:


ser.loc['AAPL']


# In[17]:


ser.loc['AAPL','2016-10-03']


# In[18]:


ser.loc[:,'2016-10-03']


# In[19]:


stocks.set_index(['Symbol','Date'],inplace=True)


# In[20]:


stocks


# In[21]:


stocks.sort_index(inplace=True)


# In[22]:


stocks


# In[29]:


stocks.loc['AAPL']


# In[33]:


stocks.loc[('AAPL','2016-10-03'),:]


# In[36]:


stocks.loc[(['AAPL','MSFT'],'2016-10-03'),:]


# In[38]:


stocks.loc[(slice(None),['2016-10-03','2016-10-04']),:]


# In[43]:


stocks.index


# In[45]:


df = pd.read_csv('http://bit.ly/drinksbycountry')


# In[46]:


df.head()


# In[47]:


df.index


# In[48]:


pd.read_table('http://bit.ly/movieusers',header=None,sep='|').head()


# In[49]:


df.set_index('country',inplace=True)
df


# In[55]:


for i in df.index:
    print(i)


# In[62]:


test = pd.read_json('pass1.json')
test


# In[63]:


people = {
    'first' : ['Corey', 'Jane', 'John'],
    'last' : ['Schafer', 'Doe', 'Doe'],
    'email' : ['CoreySchafer@gmail.com','JaneDoe@gmail.com','JohnDoe@gmail.com']
}


# In[65]:


import json
people = json.dumps(people)
loaded_people = json.loads(people)


# In[66]:


loaded_people


# In[68]:


test1 = pd.read_json('pass1 (1).json')


# In[69]:


test1


# In[2]:


import pandas as pd
df = pd.read_csv('gapminder.tsv.txt',sep='\t')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.loc[[0,1,2]]


# In[11]:


df.loc[[0,1,2]]['country']


# In[13]:


df.loc[[0,1,2],['country','pop']]


# In[14]:


df[df.country == 'United States']


# In[17]:


import numpy as np
df.groupby(['year','continent'])[['lifeExp','gdpPercap']].agg(np.mean).reset_index()


# In[18]:


import seaborn as sns
tips = sns.load_dataset('tips')
tips


# In[20]:


tips.loc[(tips.smoker == 'No') & (tips.total_bill >= 10)]


# In[22]:


tips.groupby(['smoker','day','time'])['total_bill'].mean()


# In[37]:


# TIDY DATA:
# 1)Every variable is a column
# 2)Every row is an observation
# 3)Every type of observational unit forms a table


# In[38]:


# 5 major problems that exist in a messy dataset:
# 1)Column headers are values, not variable names
# 2)multiple variables are stored in one column
# 3)variables stored in both rows and columns
# 4)multiple types of observational units are stored in the same table
# 5)a single observational unit is stored in multiple tables.


# In[23]:


pew = pd.read_csv('pew.csv')
pew
# 1)columns containing values, not variables


# In[24]:


pew.groupby('<$10k').mean()


# In[30]:


pew_copy = pew.melt(id_vars='religion')
# id_vars => column which wont be changed, anything not specified in id_vars, autom goes into value_vars
pew_copy


# In[31]:


pew_copy.groupby('variable').mean()


# In[32]:


pew.melt(id_vars='religion',var_name='income',value_name='count')


# In[35]:


billboard = pd.read_csv('billboard.csv')
billboard


# In[36]:


billboard.melt(id_vars=['year','artist','track','time','date.entered'],value_name='rank',var_name='week')


# In[41]:


bill_melt = billboard.melt(id_vars=['year','artist','track','time','date.entered'],value_name='rank',var_name='week')
bill_melt.groupby('artist')['rank'].mean().sort_values(ascending=True)


# In[42]:


# 2)multiple variables stored in one column:
ebola = pd.read_csv('country_timeseries.csv')
ebola


# In[44]:


eb_long = ebola.melt(id_vars=['Date','Day'],var_name='cd_country',value_name='count')
# melt is the reverse operation of pivot.
# melt creates long dataset from wide, while pivot does the opposite
eb_long


# In[49]:


eb_split = eb_long['cd_country'].str.split('_',expand=True)
eb_split
# .str is the string accessor, allows us to access all the string methods of python, but allow us to work on column
# 2).dt
# 3).cat


# In[50]:


eb_long[['status','country']] = eb_split
eb_long


# In[51]:


# 3)variables are stored in both rows and columns:
weather = pd.read_csv('weather.csv')
weather


# In[56]:


weather_long = weather.melt(id_vars=['id','year','month','element'],var_name='day',value_name='temp')
weather_long


# In[53]:


# to solve the third problem, use pivot, pivot_table:
# 1)pivot does not handle duplicate values, but in pivot_table there is an attribute called 'aggfunc', which by default takes the mean of duplicate values.
# for example in the weather dataset if for a particular id, year, month, day, there are two observations of tmax, pivot_table would take mean of those two values by default, but pivot wont handle duplicate values.


# In[60]:


weather_long.pivot_table(index=['id','year','month','day'],columns='element',values='temp',dropna=False).reset_index()


# In[62]:


weather_long.pivot(index=['id','year','month','day'],columns='element',values='temp').reset_index()


# In[63]:


1000000000000


# In[64]:


1_000_000_000_000


# In[65]:


tbl1 = pd.read_csv('table1.csv')
tbl1


# In[66]:


tbl2 = pd.read_csv('table2.csv')
tbl2


# In[71]:


tbl2.pivot_table(index=['country','year'],columns='type',values='count').reset_index()


# In[67]:


tbl3 = pd.read_csv('table3.csv')
tbl3


# In[76]:


tbl3['pop'] = tbl3['rate'].str.split('/',expand=True)[1]


# In[79]:


tbl3['rate'].str.split('/').str.get(1)
# another way to get the pop column


# In[77]:


tbl3


# In[80]:


df = pd.DataFrame({
    'a': [10,20,30],
    'b': [20,30,40]
})
df


# In[81]:


df['a'] ** 2


# In[82]:


def my_sq(x):
    return x**2


# In[84]:


df['a'].apply(my_sq)


# In[85]:


def my_exp(x,e):
    return x**e


# In[89]:


df.a.apply(my_exp,e=4)


# In[90]:


def print_me(x):
    print(x)


# In[92]:


df.apply(print_me)
# an entire column is passed as the first argument, not individual values of each column, once one column is complete, then another column is passed.


# In[93]:


def avg_3(x,y,z):
    return x+y+z/3


# In[95]:


df.apply(avg_3)
# returns an error


# In[96]:


def avg_3_np(col):
    return np.mean(col)


# In[97]:


df.apply(avg_3_np)


# In[102]:


# if we have to implement the same with the first approach:
def avg_3(col):
    x = col[0]
    y = col[1]
    z = col[2]
    print(x,y,z)
    return (x+y+z)/3


# In[103]:


df.apply(avg_3)


# In[104]:


def avg_2_mod(x,y):
    if x==20:
        return np.NaN
    else:
        return (x+y)/2


# In[107]:


avg_2_mod(df['a'],df['b'])


# In[109]:


avg_2_mod_vec = np.vectorize(avg_2_mod)


# In[110]:


avg_2_mod_vec(df['a'],df['b'])


# In[111]:


@np.vectorize
def avg_2_mod(x,y):
    if x==20:
        return np.NaN
    else:
        return (x+y)/2


# In[112]:


avg_2_mod(df['a'],df['b'])


# In[113]:


import numba


# In[116]:


tbl3.drop(columns='pop',inplace=True)


# In[118]:


tbl3


# In[126]:


def get_p(x):
#     print(x)
    pop = x.split('/')[1]
    return pop


# In[127]:


tbl3.rate.apply(get_p)


# In[128]:


import seaborn as sns
tips = sns.load_dataset('tips')
tips


# In[130]:


import matplotlib.pyplot as plt
tips.tip.plot(kind='hist')


# In[132]:


tips.smoker.value_counts().plot(kind='bar')


# In[133]:


sns.countplot(tips.smoker)


# In[134]:


sns.distplot(tips.total_bill)
# for continuous variables


# In[141]:


sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',fit_reg=False,col='smoker',row='day')


# In[143]:


titanic = sns.load_dataset('titanic')
titanic


# In[148]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,8))
sns.distplot(titanic.fare,ax=ax1)
sns.boxplot(x='class',y='fare',data=titanic,ax=ax2)


# In[2]:


import pandas as pd
df = pd.DataFrame([('Foreign Cinema','Restaurant',289.0),
                  ('Liho Liho','Restaurant',224.0),
                  ('500 Club','bar',80.5),
                  ('The Square','bar',25.30)],
                 columns=('name','type','AvgBill'))
df


# In[3]:


df.loc[2,'AvgBill']


# In[4]:


df.iloc[2,2]


# In[5]:


df.at[2,'AvgBill']


# In[6]:


df.at[2,'AvgBill'] = 101


# In[7]:


df


# In[8]:


df.loc[2,'AvgBill'] = 1010
df


# In[9]:


df.iloc[2,2] = 10101
df


# In[10]:


df.iat[2,2]


# In[11]:


df.iat[2,2] = 101010


df


# In[3]:


import pandas as pd
df1 = pd.read_excel('corruption.xlsx')
df1.head()


# In[4]:


df2 = pd.read_csv('ict goods exports.csv')
df2.head()


# In[7]:


df2 = df2[3:]
df2.head()


# In[12]:


df2.set_index('Data Source',inplace=True)
df2.head()


# In[9]:


df2.reset_index()


# In[11]:


df1.set_index('Economy',inplace=True)


# In[ ]:


df2.set_index('')


# In[2]:


from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[3]:


pip install arch


# In[3]:


from datetime import datetime, timedelta


# In[4]:


start = datetime(2005,3,1)
end = datetime(2021,1,12)


# In[5]:


df = pd.read_excel('Book1.xlsx')
df.head()


# In[6]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(df.SensexReturns)


# In[7]:


plt.figure(figsize=(10,4))
plt.plot(df.NiftyReturns)


# In[8]:


df.set_index('Date',inplace=True)


# In[9]:


plot_pacf(df.SensexReturns)
plt.show()


# In[11]:


df.dropna(inplace=True)
model = arch_model(df.SensexReturns,p=1,q=1)
model_fit = model.fit()


# In[12]:


df.dropna(inplace=True)
model1 = arch_model(df.NiftyReturns,p=1,q=1)
model_fit1 = model1.fit()


# In[15]:


model_fit1.summary()


# In[7]:


model_fit.summary()


# In[21]:


df.SensexReturns[:-365]


# In[13]:


rolling_predictions1 = []
test_size = 365

for i in range(test_size):
    train1 = df.NiftyReturns
    model1 = arch_model(train1, p=1, q=1)
    model_fit1 = model1.fit(disp='off')
    pred1 = model_fit1.forecast(horizon=1)
    rolling_predictions1.append(np.sqrt(pred1.variance.values[-1,:][0]))


# In[15]:


rolling_predictions1 = pd.Series(rolling_predictions1)


# In[16]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
preds, = plt.plot(rolling_predictions1)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)


# In[16]:


rolling_predictions1 = []
test_size = 365

for i in range(test_size):
    train1 = df.NiftyReturns[:-(test_size-i)]
    model1 = arch_model(train1, p=1, q=1)
    model_fit1 = model1.fit(disp='off')
    pred1 = model_fit1.forecast(horizon=1)
    rolling_predictions1.append(np.sqrt(pred1.variance.values[-1,:][0]))


# In[17]:


rolling_predictions1 = pd.Series(rolling_predictions1, index=df.NiftyReturns.index[-test_size:])


# In[8]:


rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = df.SensexReturns[:-(test_size-i)]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


# In[9]:


rolling_predictions = pd.Series(rolling_predictions, index=df.SensexReturns.index[-test_size:])


# In[28]:


rolling_predictions


# In[11]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
true, = plt.plot(df.SensexReturns[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[18]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
true, = plt.plot(df.NiftyReturns[-test_size:])
preds, = plt.plot(rolling_predictions1)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[18]:


df.NiftyReturns[:-(730)]


# In[25]:


df.SensexReturns[-730:-480]


# In[35]:


rolling_predictions1 = []
test_size = 250

for i in range(test_size):
    train1 = df.NiftyReturns[:-(test_size-i)]
    model1 = arch_model(train1, p=1, q=1)
    model_fit1 = model1.fit(disp='off')
    pred1 = model_fit1.forecast(horizon=1)
    rolling_predictions1.append(np.sqrt(pred1.variance.values[-1,:][0]))


# In[36]:


rolling_predictions1 = pd.Series(rolling_predictions1, index=df.NiftyReturns.index[-730:-480])


# In[37]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
true, = plt.plot(df.NiftyReturns[-730:-480])
preds, = plt.plot(rolling_predictions1)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[48]:


rolling_predictions1 = []
test_size = 250

for i in range(test_size):
    train1 = df.SensexReturns[:-(730-i)]
    model1 = arch_model(train1, p=1, q=1)
    model_fit1 = model1.fit(disp='off')
    pred1 = model_fit1.forecast(horizon=1)
    rolling_predictions1.append(np.sqrt(pred1.variance.values[-1,:][0]))


# In[52]:


df.SensexReturns[:-(730)]


# In[49]:


rolling_predictions1 = pd.Series(rolling_predictions1, index=df.SensexReturns.index[-730:-480])


# In[50]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
true, = plt.plot(df.SensexReturns[-730:-480])
preds, = plt.plot(rolling_predictions1)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[53]:


df.SensexReturns[-730:-480]


# In[54]:


rolling_predictions1 = []
test_size = 250

for i in range(test_size):
    train1 = df.NiftyReturns[:-(730-i)]
    model1 = arch_model(train1, p=1, q=1)
    model_fit1 = model1.fit(disp='off')
    pred1 = model_fit1.forecast(horizon=1)
    rolling_predictions1.append(np.sqrt(pred1.variance.values[-1,:][0]))


# In[55]:


rolling_predictions1 = pd.Series(rolling_predictions1, index=df.NiftyReturns.index[-730:-480])


# In[56]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
true, = plt.plot(df.NiftyReturns[-730:-480])
preds, = plt.plot(rolling_predictions1)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[ ]:




