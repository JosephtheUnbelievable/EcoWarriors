import xgboost as xgb
import pandas as pd
import os    
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable line wrapping
pd.set_option('display.max_colwidth', None)  # Show full column content

paths = ['September.csv',
         'October.csv',
         'November.csv',
         'December.csv',
         'January.csv',
         'February.csv']

columns = ['Day','Weekday','Main Dish','Dish 1','Dish 2','Dish 3','Dish 4',
           'Soup','Inspector(s)','Measured Weight (kg)',
           'Type of Cuisine','Contains Soup?','Has Fried Food?',
           'Has Rice?','Has Sauce?','Has Chicken?','Has Pork?','Has Eggs?']
drop_columns = ['Day','Weekday','Main Dish','Dish 1','Dish 2','Dish 3','Dish 4',
           'Soup','Inspector(s)']
#-------------------------------Data manipulating
df = pd.DataFrame()
for path in paths:
    df = pd.concat([df,pd.read_csv(path)],axis = 0)
for columns in drop_columns:
    df.drop(columns,axis=1,inplace=True)

for column in df.columns:
    if(column == 'Measured Weight (kg)'):
        continue
    df[column] = df[column].replace({'True': True, 'False': False})
    df[column] = df[column].astype(bool)
print(df['Contains Soup?'].dtype)

df.dropna(subset=['Measured Weight (kg)'],inplace=True)
df.reset_index(drop=True,inplace=True)

df = df.sample(frac=1).reset_index(drop=True) #shuffling
#--------------------------------------------------------

N = int(df.shape[0]*0.85) #training size
print(N)
X = df[['Contains Soup?','Has Fried Food?',
           'Has Rice?','Has Sauce?','Has Chicken?','Has Pork?','Has Eggs?']]
Y = df[['Measured Weight (kg)']]

#training = 85%, testing = 15%
X_train = X.iloc[:N].copy()
Y_train = Y.iloc[:N].copy()

X_test = X.iloc[N:].copy()
Y_test = Y.iloc[N:].copy()


print(X_train)
print(Y_train)

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)

params = {
    'objective':'reg:squarederror',
    'eval_metric':'rmse',
    'max_depth':6,
    'eta':0.2,
    'silent':1
}

num_round=20
watchlist = [(dtrain, 'train')]
bst = xgb.train(params,dtrain,num_round,evals=watchlist,verbose_eval=True)
predictions = bst.predict(dtest)
print("Predictions on the test data (food waste weight in kg):")
print(predictions)

# Optionally, compare predictions to actual values
print("\nActual food waste weights (kg):")
print(pd.concat([Y_test,X_test],axis=1))
