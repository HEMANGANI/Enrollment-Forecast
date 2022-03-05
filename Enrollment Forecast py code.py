# -*- coding: utf-8 -*-

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from google.colab import drive
#drive.mount('/content/drive')

# read data from file
test_data = pd.read_csv("test data.csv")
train_data = pd.read_csv("train data.csv")

train_data.head()

test_data.head()

train_data['Campus_Id'] = train_data['Campus_Id'].astype('object')
test_data['Campus_Id'] = test_data['Campus_Id'].astype('object')
train_data.info()

categorical_columns_train = train_data.select_dtypes(exclude = 'int64').columns
categorical_columns_test = test_data.select_dtypes(exclude = 'int64').columns
print(len(categorical_columns_train),len(categorical_columns_test))
print(categorical_columns_test)

categories_proceed = []
for i in categorical_columns_train:
    unique_program_train = np.unique(train_data[i])
    unique_program_test = np.unique(test_data[i])
    print(i, len(unique_program_train),len(unique_program_test))
    if(len(unique_program_train)==len(unique_program_test)):
        categories_proceed.append(i)
        
print(categories_proceed)

print((np.unique(train_data['Campus_Id'])),(np.unique(test_data['Campus_Id'])))
print((np.unique(train_data['Attendance'])),(np.unique(test_data['Attendance'])))

#converting all categories to ONE HOT ENCODING: Train Data
from sklearn.preprocessing import OneHotEncoder
for i in categories_proceed:
    onehot_features_train = pd.get_dummies(train_data[i],drop_first=True)
    train_data = pd.concat([pd.DataFrame(train_data), onehot_features_train], axis=1)
    train_data = train_data.drop(i, axis=1)
    
train_data.rename(columns={'2':'fg_2', '3':'fg_3', '4':'fg_4', '5':'fg_5'}, inplace=True)
pd.set_option('display.max_columns', None)
train_data.head()

#converting all categories to ONE HOT ENCODING: Test Data
for i in categories_proceed:
    onehot_features_test = pd.get_dummies(test_data[i],drop_first=True)
    test_data = pd.concat([pd.DataFrame(test_data), onehot_features_test], axis=1)
    test_data = test_data.drop(i, axis=1)
    
test_data.rename(columns={'2':'fg_2', '3':'fg_3', '4':'fg_4', '5':'fg_5'}, inplace=True)
pd.set_option('display.max_columns', None)
test_data.head()

print(train_data.shape, test_data.shape)

#converting to ONE HOT ENCODING: Train Data: Campus_Id
cid_onehot_features = pd.get_dummies(train_data['Campus_Id'], drop_first=True)
train_data = pd.concat([pd.DataFrame(train_data), cid_onehot_features], axis=1)
train_data = train_data.drop('Campus_Id', axis=1)
train_data.rename(columns={1:'cid_1', 2:'cid_2', 3:'cid_3', 4:'cid_4'}, inplace=True)
train_data.head()

#converting to ONE HOT ENCODING: Test Data: Campus_Id
cid_onehot_features = pd.get_dummies(test_data['Campus_Id'], drop_first=True)
test_data = pd.concat([pd.DataFrame(test_data), cid_onehot_features], axis=1)
test_data = test_data.drop('Campus_Id', axis=1)
test_data.insert(35,'add_col',0)
test_data.rename(columns={1:'cid_1', 2:'cid_2', 'add_col':'cid_3', 4:'cid_4'}, inplace=True)
test_data.head()

print(train_data.shape, test_data.shape)

#converting 3 categories to ONE HOT ENCODING: Train Data: Attendance
attd_onehot_features = pd.get_dummies(train_data['Attendance'])
attd_dummy_features = attd_onehot_features.iloc[:,:-1]
train_data = pd.concat([pd.DataFrame(train_data), attd_dummy_features], axis=1)
train_data = train_data.drop('Attendance', axis=1)
train_data.head()

#converting 3 categories to ONE HOT ENCODING: Test Data: Attendance
attd_onehot_features = pd.get_dummies(test_data['Attendance'])
#attd_dummy_features = attd_onehot_features.iloc[:,:-1] beacuse it has 2 attribute in test dataset
test_data = pd.concat([pd.DataFrame(test_data), attd_onehot_features], axis=1)
test_data = test_data.drop('Attendance', axis=1)
test_data.head()

print(train_data.shape, test_data.shape)

dateis = np.array(train_data['Fiscal Year'])
for i in range(len(dateis)):
    #print(dateis[i])
    dateis[i] = dateis[i].replace('/','')
    dateis[i] = dateis[i][2:]

train_data['Fiscal Year'] =  dateis
train_data['Fiscal Year'] = train_data['Fiscal Year'].astype(int)
train_data.head()

dateis = np.array(test_data['Fiscal Year'])
for i in range(len(dateis)):
    #print(dateis[i])
    dateis[i] = dateis[i].replace('/','')
    dateis[i] = dateis[i][2:]

test_data['Fiscal Year'] =  dateis
test_data['Fiscal Year'] = test_data['Fiscal Year'].astype(int)
test_data.head()

# Program Grouping: 147 to 8 features
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=8, input_type='string')
hashed_features = fh.fit_transform(train_data['Program Grouping'])
hashed_features = hashed_features.toarray()
hashed_features.shape
train_data=pd.concat([pd.DataFrame(train_data),pd.DataFrame(hashed_features)],axis=1)
train_data.rename(columns={0:'pg_0', 1:'pg_1', 2:'pg_2', 3:'pg_3', 4:'pg_4', 5:'pg_5', 6:'pg_6', 7:'pg_7'}, inplace=True)
train_data = train_data.drop('Program Grouping', axis=1)
train_data.head()

# Program Grouping: 147 to 8 features: TEST
fh = FeatureHasher(n_features=8, input_type='string')
hashed_features = fh.fit_transform(test_data['Program Grouping'])
hashed_features = hashed_features.toarray()
hashed_features.shape
test_data=pd.concat([pd.DataFrame(test_data),pd.DataFrame(hashed_features)],axis=1)
test_data.rename(columns={0:'pg_0', 1:'pg_1', 2:'pg_2', 3:'pg_3', 4:'pg_4', 5:'pg_5', 6:'pg_6', 7:'pg_7'}, inplace=True)
test_data = test_data.drop('Program Grouping', axis=1)
test_data.head()

print(train_data.shape, test_data.shape)

X = train_data.drop('Unique Headcount', axis=1)
X = X.drop('Id', axis=1)
Y = train_data['Unique Headcount']

print(X.shape, Y.shape, test_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
list(map(lambda x: x.shape, [X, Y, x_train, x_test, y_train, y_test]))

# Instantiate model with 100 decision trees
from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor
clf = XGBRegressor(random_state = 42,learning_rate=0.5, max_depth=8, n_estimators=120)

clf.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
import math
# Use the forest's predict method on the test data
predictions = clf.predict(x_test)
predictions = abs(predictions)
predictions = np.round(predictions)

# model evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

test_run = test_data.drop('Id', axis=1)
result = clf.predict(test_run)
result = abs(result)
result = np.round(result)
result = pd.DataFrame(result,columns=['Unique Headcount'])
#result.head()

result['Id'] = test_data['Id']
result = result[['Id','Unique Headcount']]
result['Id'] = result['Id'].astype(int)
result['Unique Headcount'] = result['Unique Headcount'].astype(int)
result.head(20)

print(result.shape,test_data.shape)

result.to_csv("result.csv", index=False)
