# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here

dataset = pd.read_csv(path)
print(dataset.head())
print(dataset['Soil_Type34'].value_counts())
dataset = dataset.drop(['Id'],1)
print(dataset.describe())






















# read the dataset



# look at the first five columns


# Check if there's any column which is not useful and remove it like the column id


# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 

cols = dataset.columns
size = len(cols)-1
x = dataset['Cover_Type']
y = dataset.drop(['Cover_Type'],1)
for i in range(size):
    sns.violinplot(y = y.iloc[:,i])


















#number of attributes (exclude target)


#x-axis has target attribute to distinguish between classes


#y-axis shows values of an attribute


#Plot violin for all attributes



# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here

subset_train = y.iloc[:,0:10]
data_corr = subset_train.corr()
sns.heatmap(data_corr,annot=True)
correlation = data_corr.unstack().sort_values(kind='quicksort')
# print(correlation[(correlation>upper_threshold) | (correlation != 1)] )
corr_var_list = correlation[(abs(correlation)>upper_threshold) & (correlation < 1) ]
# corr_var_list = set(corr_val_list.index)

print(corr_var_list)




# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X = dataset.drop(['Cover_Type'],1)
Y = dataset['Cover_Type']
X_train, X_test,y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train)
cols1 = X_train.columns
cols2 = X_test.columns
X_test_temp = scaler.fit_transform(X_test)
# X_train1 = pd.concat(pd.DataFrame(X_train_temp), X_train)
# X_test1 = pd.concat(pd.FataFrame(X_test_temp),X_test)
scaled_features_train_df = pd.DataFrame(data = X_train_temp, columns=cols1,index = X_train.index)
scaled_features_test_df = pd.DataFrame(data =X_test_temp , columns=cols2 ,index = X_test.index)
print(scaled_features_train_df.head())

# X_train1 = 








# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.



#Standardized
#Apply transform only for continuous data


#Concatenate scaled continuous data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:


skb = SelectPercentile(score_func = f_classif, percentile = 20)
# print(scaled_features_train_df)
print(Y_train.shape)
predictors = skb.fit_transform(scaled_features_train_df,Y_train)
# print(scaled)
scores = list(skb.scores_)
# print(scores)
Features = X_train.columns
d = {'Features': Features, 'scores': scores}
dataframe = pd.DataFrame(data = d)
# dataframe['Feature'] = Features
# dataframe['scores'] = scores
dataframe = dataframe.sort_values(by = ['scores','Features'],ascending = False)
print(dataframe)
top_k_predictors = list(dataframe['Features'].iloc[0:11])
print(top_k_predictors)







# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())
model_fit_all_features = clf1.fit(X_train,Y_train)
predictions_all_features = model_fit_all_features.predict(X_test)
score_all_features = accuracy_score(Y_test,predictions_all_features)
print(Y_test.shape)
print(predictions_all_features.shape)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
print(scaled_features_test_df[top_k_predictors].shape)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
print(Y_test.shape)
print(predictions_top_features.shape)
score_top_features = accuracy_score(Y_test,predictions_top_features)
print(score_all_features)
print(score_top_features)




