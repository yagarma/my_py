#!/usr/bin/env python
# coding: utf-8

# # Pandas

# Dieses Notebook ist ein Auszug von: https://github.com/mjbahmani/10-steps-to-become-a-data-scientist

# In[1]:


import numpy as np
import pandas as pd


# <img src='https://www.kdnuggets.com/wp-content/uploads/pandas-02.png'>
# [Image Credit](https://www.kdnuggets.com/2017/01/pandas-cheat-sheet.html)

# In[2]:


pd.__version__


# ## Wie erzeuge ich eine Pandas_Serie

# In[3]:


dict_1 = {'Name': 'Masha', 'Item purchased': 'cat food', 'Cost': 19.85}
print(dict_1)
type(dict_1)


# In[4]:


series_1 = pd.Series(dict_1)
print(series_1)


# In[5]:


series_1.index


# ## Wie erzeuge ich einen Pandas-DataFrame

# In[6]:


#Erzeugen eines DataFrames
#Beispiel: Series
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df_1 = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
#df.head()


# In[7]:


print(df_1)
print(type(df_1))


# In[8]:


df_1


# In[9]:


type(df_1)


# In[10]:


df_1.index


# In[11]:


df_1.columns


# In[12]:


df_1.values


# In[13]:


df_1.head()


# In[14]:


purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00, 'Test': 55})
df_2 = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df_2.head()


# ### Erstes Untursuchungen von DataFrames

# In[15]:


df_2.head(2)


# In[16]:


df_2.tail(2)


# In[17]:


df_2.describe() # für alle numerische Daten


# In[18]:


get_ipython().run_line_magic('pinfo', 'df_2.describe')


# In[19]:


df_2.describe(include = 'all')


# In[20]:


df_2.info()


# ##  Erstes Zugreifen

# In[21]:


df.loc['Store 1', 'Name']


# In[22]:


df.loc['Store 1','Cost'] # loc for named index


# In[23]:


df.iloc[0:,0:2] # iloc for counting index(automatic)


# In[24]:


df_2.loc['Store 1', ['Name', 'Cost']]


# In[25]:


df_2.loc['Store 1', ['Name']]


# In[26]:


df.T


# In[27]:


df.T.index


# In[28]:


df.loc['Store 1'].iloc[:,1:]


# In[29]:


get_ipython().run_line_magic('pinfo', 'df.drop')


# In[30]:


df


# In[31]:


df_3 = df.drop('Cost', axis=1).copy()
df_3


# In[32]:


df


# In[33]:


df.drop('Store 1')


# In[34]:


copy_df = df.copy()
print(copy_df)
del copy_df['Name']
copy_df


# In[35]:


df


# In[36]:


costs = df['Cost']
costs
costs += 2 # Ändert den DataFrame Dateien
df


# ## Importieren und "Abfragen"

# In[37]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[38]:


df = pd.read_csv('98_Daten/trainh.csv')


# In[39]:


df.tail(5)


# In[40]:


df.head(10)


# In[41]:


df.describe


# In[42]:


df.info()


# In[43]:


df.info()


# In[44]:


df.describe()


# In[45]:


pd.set_option('display.max_columns', 500)


# In[46]:


df_csv_aus_html = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')


# In[47]:


df_csv_aus_html


# In[48]:


df_csv_aus_html.describe()


# ## Fancy Indexing

# In[49]:


df['SalePrice'] > 50000


# In[50]:


only_SalePrice = df.where(df['SalePrice'] > 180000)
only_SalePrice


# In[51]:


only_SalePrice_2 = only_SalePrice.dropna()
only_SalePrice_2.head()


# In[52]:


only_SalePrice = df.where((df['SalePrice'] > 180000) | (df['SalePrice'] < 130000) | (df['HouseStyle'] == '1Story'))
only_SalePrice


# ###  Pandas zum Bereinigen von fehlenden Werten

# In[53]:


df = pd.read_csv('98_Daten/traint.csv')
df.head()


# In[54]:


df.info()


# In[55]:


df.isnull().head(10)


# In[56]:


print(df.isnull().sum())


# In[57]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame.fillna')


# In[58]:


print(df['Age'].mean())
df['Age'] = df['Age'].fillna(df['Age'].mean())
print(df.isnull().sum())


# ## EDA - Gruppieren und Pivotieren von Daten mit Pandas

# In[59]:


import seaborn as sns
titanic = sns.load_dataset('titanic')


# In[60]:


titanic


# In[61]:


titanic.groupby('sex')[['survived']].mean()


# In[62]:


titanic.groupby('class')[['survived']].mean()


# In[63]:


df.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack()


# In[64]:


titanic.groupby(['deck', 'pclass'])['survived'].aggregate('mean').unstack()


# In[65]:


# In Welchem Verzeichnis befinde ich mich gerade?
get_ipython().run_line_magic('pwd', '')


# In[66]:


get_ipython().run_line_magic('ls', '')


# In[67]:


df.head(3)


# In[77]:


df[df['Age']<40].groupby('Sex')[['Survived']].mean()


# In[70]:


df.groupby(['Sex', 'Pclass'])['Survived'].aggregate(['mean']).unstack()


# Pandas erlaubt uns ganz einfaches Verknüpfen von Tabellen. Ganz ähnlich zu einem JOIN in SQL. Am schnellsten funktioniert das, wenn wir den Index ausnutzen. (Es geht aber natürlich auch ohne, dass wir den Index verwenden...)

# In[85]:


df[df['Age']<40].groupby(['Sex', 'Pclass'])[['Age', 'Survived']].mean()


# In[89]:


titanic[titanic['age']>1].groupby(['sex', 'pclass', 'alone'])[['age', 'survived']].mean()


# ## Merging (Joins) & Concat (Anhängen)

# <img src='https://static1.squarespace.com/static/54bb1957e4b04c160a32f928/t/5724fd0bf699bb5ad6432150/1462041871236/?format=750w'>
# [Image Credit](https://www.ryanbaumann.com/blog/2016/4/30/python-pandas-tosql-only-insert-new-rows)

# In[90]:


# Wie lege ich nochmal einen DataFrame an, wenn ich gerade keine Datei importieren möchte/kann?
df = pd.DataFrame([{'Name': 'MJ', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df


# In[91]:


#Was tue ich, wenn ich mal eine Spalte vergessen habe?
df['Date'] = ['December 1', 'January 1', 'mid-May']
df


# In[92]:


#So kann man natürlich jeden Datentyp einfügen
df['Delivered'] = True
df


# In[93]:


df['Feedback'] = ['Positive', None, 'Negative']
df


# In[94]:


#Wie kann ich denn den Index anpassen?
adf = df.reset_index()

#reset_index erzeugt einen neuen, fortlaufenden Index
#set_index erlaubt es, eine beliebige Spalte zum Index zu machen.

adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf


# ## Und wie könnte ich jetzt DataFrames (Tabellen) miteinander verknüpfen?
# Zuerst benötigt man zwei Tabellen mit einem "Key", über den sie sich verbinden lassen.

# In[95]:


## Beispiel

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')

student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')


# In[96]:


student_df


# In[97]:


staff_df


# ##### Die Pandas-Funktion "merge" erlaubt es uns nun, die Tabellen zu verknüpfen

# In[99]:


pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)


# In[100]:


pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)


# In[101]:


pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)


# In[102]:


pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)


# In[106]:


pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)


# ## Geht das denn auch ohne den Index? --> Ja, aber etwas langsamer!

# In[107]:


staff_df = staff_df.reset_index()
student_df = student_df.reset_index()


# In[108]:


staff_df


# In[109]:


student_df


# In[110]:


pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# ## Die Tabellen können dabei natürlich beliebig groß sein...

# In[111]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])

student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])

pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# In[112]:


staff_df


# In[113]:


student_df


# ### ... Und man kann natürlich auch mehrere Felder als Verknüpfungsbedingungen übergeben

# In[114]:


staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])


# In[116]:


staff_df


# In[117]:


student_df


# In[118]:


pd.merge(staff_df, student_df, how='left', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])


# In[119]:


pd.merge(staff_df, student_df, how='outer', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])


# ## Wie kann ich denn Tabellen verbinden, die gleichartige Informationen zu unterschiedlichen Dingen oder Personen enthalten?
# (Wenn es also keinen Key gibt?) (Im SQL wäre das bspw. ein UNION)

# In[120]:


# Creating first dataframe 
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
                    'B': ['B0', 'B1', 'B2', 'B3'], 
                    'C': ['C0', 'C1', 'C2', 'C3'], 
                    'D': ['D0', 'D1', 'D2', 'D3']}, 
                    index = [0, 1, 2, 3]) 
  
# Creating second dataframe 
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'], 
                    'B': ['B4', 'B5', 'B6', 'B7'], 
                    'C': ['C4', 'C5', 'C6', 'C7'], 
                    'D': ['D4', 'D5', 'D6', 'D7']}, 
                    index = [4, 5, 6, 7]) 
  
# Creating third dataframe 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'], 
                    'B': ['B8', 'B9', 'B10', 'B11'], 
                    'C': ['C8', 'C9', 'C10', 'C11'], 
                    'D': ['D8', 'D9', 'D10', 'D11']}, 
                    index = [8, 9, 10, 11]) 


# In[122]:


df1


# In[123]:


df2


# In[124]:


df3


# In[127]:


get_ipython().run_line_magic('pinfo', 'pd.concat')


# In[126]:


# Concatenating the dataframes 
pd.concat([df1, df2, df3])


# In[128]:


pd.concat([df1, df2, df3], axis=1)


# In[134]:


df2.columns = ['C','D','E','F']


# In[132]:


df1.columns = ['A','B','C','D']


# In[135]:


pd.concat([df1, df2, df3])


# ## Indizieren von DataFrames & Explorative Datenanalyse

# In[137]:


df = pd.read_csv('98_Daten/traint.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Zeitfunktionalitäten in Pandas (Timestamps)

# In[138]:


t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1


# In[144]:


get_ipython().run_line_magic('pinfo', 'pd.Timestamp')


# In[140]:


type(t1.index)


# In[141]:


t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2


# In[142]:


type(t2.index)


# In[143]:


get_ipython().run_line_magic('pinfo', 'pd.Period')


# In[145]:


#Daten in Zeitstempel konvertieren
d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3


# In[146]:


get_ipython().run_line_magic('pinfo', 'pd.to_datetime')


# In[149]:


ts3.index = pd.to_datetime(ts3.index)
ts3


# In[148]:





# In[156]:


datum = pd.to_datetime('4.7.12', dayfirst=True)
print(datum)


# In[154]:


type(datum)


# In[159]:


pd.Timestamp('9/4/2016')-pd.Timestamp('9/5/2016')


# In[160]:


pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')


# In[179]:


my_date = pd.to_datetime('10.01.2020', format='%d.%m.%Y')


# In[180]:


dates = pd.date_range(my_date, periods=9, freq='2W-SUN')
dates


# In[164]:


import datetime
datetime.datetime.now()


# In[171]:


get_ipython().run_line_magic('pinfo', 'pd.date_range')


# In[174]:





# ## PIVOT

# In[185]:


import seaborn as sns
tips = sns.load_dataset("tips")
tips.head(10)


# In[191]:


pd.pivot_table(tips, values='total_bill', 
                     index=['sex'], 
                     columns=['time','smoker'], aggfunc='sum')


# In[192]:


pd.pivot_table(tips, values='tip', 
                     index=['sex'], 
                     columns=['time','smoker'], aggfunc='sum')


# In[190]:


get_ipython().run_line_magic('pinfo', 'pd.pivot')


# ## Melt

# In[188]:


df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},'B': {0: 1, 1: 3, 2: 5},'C': {0: 2, 1: 4, 2: 6}})
df


# In[189]:


get_ipython().run_line_magic('pinfo', 'pd.melt')


# In[193]:


pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])


# In[ ]:


f

