import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
np.random.seed(42)

pd.set_option('display.max_columns', None)
df=pd.read_csv('audio_data.csv')

# use pandas get_dummies function to one hot encode the "label" column
one_hot_encoded = pd.get_dummies(df['label'])

# concatenate the one hot encoded columns with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

# drop the original "Gender" column
df.drop('label', axis=1, inplace=True)
df = df.drop('female',axis=1)

# from pandas_profiling import ProfileReport
# report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
# report

x = df[['meanfun' ,'sd','median','Q25','Q75','IQR','skew','kurt','maxfun','minfun','mode']].values
y=df['male']
# Standerize the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaler.fit(x)
x = scaler.transform(x)
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.12)

from sklearn import svm
model=svm.SVC(kernel='linear',max_iter=1200000)
model.fit(X_train,y_train)
train_pred=model.predict(X_train)
y_preds=model.predict(X_test)
model.score(X_test , y_test)

def prediction(features):
    scaler = StandardScaler() 
    scaler.fit(features)
    x = scaler.transform(features)
    ans = model.predict(x)
    return ans[0]