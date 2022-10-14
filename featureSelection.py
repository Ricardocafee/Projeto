import numpy as np
import pandas as pd
import datetime as dt

df = pd.read_csv('noOutliers.csv')


print(df.head())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df["Date"]= pd.to_datetime(df['Date'])
df['DateStamp']=df['Date'].map(dt.datetime.timestamp)


X = df.drop(['Persons','Date'],axis=1)
Y = df['Persons']

print(X)
print(Y)

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=1)
print(fit.scores_)

list = ['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','CO2','PIR1','PIR2','DateStamp']





order = [x for _, x in sorted(zip(fit.scores_, list),reverse=True)]
orderedValues = zip(sorted(fit.scores_,reverse=True),order)

print(tuple(orderedValues))
#features = fit.transform(X)
# Summarize selected features
#print(features[0:5,:])