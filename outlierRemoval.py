from enum import auto
import numpy as np
import pandas as pd
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, make_scorer
import datetime as dt
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

def trainForOutlier():
    # Separate the classes from the train set
    
    df_classes = df['Persons']
    df_train = df.drop(['Time','Persons'], axis=1)

    print(df_train)

    # split the data into train and test 
    X_train, X_test, y_train, y_test = train_test_split(df_train, df_classes, test_size=0.30)
    print(y_test)
    #print(X_train,np.where(np.isnan(X_train)))

    # train the model on the nominal train set
    model_isf = IsolationForest().fit(X_train)

    # predict on testset
    df_pred_test = X_test.copy()

    #df_pred_test['Class'] = y_test
    df_pred_test['Pred'] = model_isf.predict(X_test)
    print(df_pred_test['Pred'])

    df_pred_test['Pred'] = df_pred_test['Pred'].map({1: 0, -1: 1})


    # measure performance
    y_pred = df_pred_test['Pred']
    print(f"Accuracy: {accuracy_score(y_pred, y_test)}\n")
    print(f"Precision: {precision_score(y_pred, y_test, average='micro')}\n")
    print(f"Recall: {recall_score(y_pred, y_test, average='micro')}\n")
    print(f"Confusion matrix:{confusion_matrix(y_pred, y_test)}\n")

df = pd.read_csv('Proj1_Dataset.csv')
#print(df.isnull().sum())
df = df.fillna(df.mean())
#print(df.isnull().sum())

df["Date"]= pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['DateStamp']=df['Date'].map(dt.datetime.timestamp)
df = df.drop(['Time'],axis=1)

print(df)

enablePlots = False

if enablePlots:
    # create histograms on all features
    df_hist = df.drop(['Date','Time'], 1)
    df_hist.hist(figsize=(20,20), bins = 50, color = "c", edgecolor='black')
    plt.show()

    # Check that features are correlated
    plt.figure(figsize=(15,4))
    f_cor = df_hist.corr()
    sns.heatmap(f_cor, cmap="Blues")
    plt.show()


    # Plot the balance of class labels
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    plt.pie(df[['Persons']].value_counts(), explode=None, labels=[0,1,2,3], autopct='%1.2f%%', shadow=False, startangle=45)
    plt.show()

# initializing the isolation forest
isolation_model = IsolationForest(contamination = 0.02)

df_train = df.drop(['Date'],axis=1)
# training the model 
isolation_model.fit(df_train)

# making predictions 
IF_predictions = isolation_model.predict(df_train)

# printing
df['outlier'] = IF_predictions
print(df)

print(df.loc[df['outlier'] == -1,['outlier']].sum())

df.plot(x='Date',y=['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','CO2','PIR1','PIR2'])
plt.show()

df.drop(df.index[df['outlier'] == -1], inplace=True)

print(df)

###################################################################################################################################################
############################## teu código

df["Overcrowded"] = df["Persons"]

df.loc[df["Persons"]>2, "Overcrowded"] = 1
df.loc[df["Persons"]<=2, "Overcrowded"] = 0

df_classes = df['Overcrowded']
df_train = df.drop(['Date','Persons','DateStamp','outlier'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(df_train, df_classes, test_size=0.15, random_state=42) 

y_train= y_train.astype('int')
y_train=np.ravel(y_train)



sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


clf = MLPClassifier(solver='lbfgs',activation='relu',random_state=1, max_iter=10000,alpha=1e-6).fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_test= y_test.astype('int')
y_pred= y_pred.astype('int')

prec = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
print("Precision: ",prec)
print("Recall: ",recall)
print("Accuracy: ",acc)
print("F1: ",f1)


plt.show()
