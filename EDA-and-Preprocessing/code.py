# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(path)

data['Rating'].hist()

data = data[data['Rating']<=5]

data['Rating'].hist()



#Code starts here


#Code ends here


# --------------
# code starts here


total_null = data.isnull().sum()

percent_null = (total_null/data.isnull().count())

missing_data = pd.concat((total_null,percent_null),axis =1,keys=['Total','Percent'])
print(missing_data)

data = data.dropna(axis=0)

total_null_1 = data.isnull().sum()

percent_null_1 = (total_null/data.isnull().count())
missing_data_1 = pd.concat((total_null_1,percent_null_1),axis =1,keys=['Total','Percent'])
print(missing_data_1)

# code ends here


# --------------

#Code starts here

sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace('\W', '')
print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].astype('int')
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
sns.regplot(x=data['Installs'],y=data['Rating'],data=data)
plt.title('Rating vs Installs [RegPlot')
#Code ends here



# --------------
#Code starts here




print(data['Price'].value_counts())

data['Price'] = data['Price'].str.replace('$','')
data['Price'] = data['Price'].astype('float')
print(data['Price'].head())
sns.regplot(x='Price',y='Rating',data=data)
plt.title('Rating vs Price [RegPlot')



#Code ends here


# --------------

#Code starts here

print(data['Genres'].value_counts())
genres = []
for x in data['Genres']:
    genres.append(x.split(';')[0])
data['Genres'] = genres

# print(data['Genres'].value_counts())

gr_mean = data[['Rating','Genres']].groupby(['Genres'],as_index=False).mean()
# print(gr_mean.head())
print(gr_mean.describe())

gr_mean = gr_mean.sort_values('Rating')
print(gr_mean.head(1))
print(gr_mean.tail(1))




#Code ends here


# --------------

#Code starts here

print(data['Last Updated'])

data['Last Updated'] = pd.to_datetime(data['Last Updated'])

max_date = data['Last Updated'].max()

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

sns.regplot(x='Last Updated Days',y='Rating',data=data)
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here


