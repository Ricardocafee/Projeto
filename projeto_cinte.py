from typing import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, make_scorer, get_scorer_names
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d, Axes3D
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, GridSearchCV, cross_validate
from sklearn import preprocessing

indexes_sensors = {
    0:"S1Temp",
    1:"S2Temp",
    2:"S3Temp",
    3:"S1Light",
    4:"S2Light",
    5:"S3Light",
    6:"PIR1",
    7:"PIR2",
    8:"Persons",
}

window_sensors = {
    0:5,         #S1Temp
    1:5,         #S2Temp
    2:5,         #S3Temp
    3:5,         #S1Light
    4:5,         #S2Light
    5:5,         #S3Light
    6: 20,       #PIR1
    7: 20,       #PIR2
}

k_sensors = {
    0:1.75,     #S1Temp
    1:2,      #S2Temp
    2:1.75,     #S3Temp
    3:1.78,     #S1Light
    4:1.78,      #S2Light
    5:1.78,     #S3Light
    6: 2,       #PIR1
    7: 2,       #PIR2
}




class average_persons:
    def __init__(self, sensor,df,k):
        self.sensor = sensor
        self.df = df
        self.mean0 = df.loc[df['Persons'] == 0, sensor].mean()
        self.std0 = df.loc[df['Persons'] == 0, sensor].std() 
        self.mean1 = df.loc[df['Persons'] == 1, sensor].mean()
        self.std1 = df.loc[df['Persons'] == 1, sensor].std() 
        self.mean2 = df.loc[df['Persons'] == 2, sensor].mean()
        self.std2 = df.loc[df['Persons'] == 2, sensor].std() 
        self.mean3 = df.loc[df['Persons'] == 3, sensor].mean()
        self.std3 = df.loc[df['Persons'] == 3, sensor].std() 
        self.k = k

def function_plot(frame):

    #for index in range(9):
        #frame.plot(x="Time",y=indexes_sensors[index])
    #df_new = frame["S1Temp"].diff(periods=70)
   # frame = frame.assign(Slope=df_new)

    #plt.scatter(x=frame["S1Light"], y = frame["Persons"])
   # plt.show()

    #plt.xlim([0,400])

    frame.plot(x="Time", y = "PIR1")
    #frame.plot(x="Time",y="PIR2")
    
    #frame.plot(x="Time", y = "Slope")
    frame.plot(x="Time", y = "Persons")


def remove_outliers(df):

    df_new = df["CO2"].diff(periods=50)
    df = df.assign(Slope_CO2=df_new)

    df = df.drop(columns=["CO2"])

    #Based on time

    average_time = np.zeros((7,len(df)))
    std_time = np.zeros((7,len(df)))  

    #Based on time

    for index in range(7):
        average_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).mean()
        std_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).std()

    sample = pd.DataFrame(df).to_numpy()
    z = np.zeros((len(df),1), dtype=object)
    sample = np.append(sample,z,axis=1)
        
    count = 0     

    average_std = [0,0,0,0,0,0,0,0]

    for i in range(len(df)):
        time = sample[i,1]
        time = int(time[:2])

        if(time<9):
            sample[i,13] = 1
        elif(time>=9 and time < 12):
            sample[i,13] = 2
        elif(time>=12 and time < 16):
            sample[i,13] = 3
        elif(time>=16 and time < 19):
            sample[i,13] = 4
        else:
            sample[i,13] = 5

        #Out of work time
        if(time<6 or time > 21):
            sample[i,5] = 0     #S1Light
            sample[i,6] = 0     #S2Light
            sample[i,7] = 0     #S3Light
            sample[i,9] = 0     #PIR1
            sample[i,10] = 0     #PIR2

        if(i>4):
            for j in range(7):

                if(sample[i,j+2] > average_time[j][i]+k_sensors[j]*std_time[j][i] or sample[i,j+2] < average_time[j][i] - k_sensors[j]*std_time[j][i]):
                    sample[i,j+2]=sample[i-1,j+2]
                    count = count+1

                
            if(sample[i,5] > 1000):       #Especific case: Date 12/01
                sample[i,5]=sample[i-1,5]  #S1Light
                #count = count+1

            if(i>=1 and i < len(df)-1):
            
                if(sample[i-1,8]==0 and sample[i+1,8]==0 and sample[i,10]==0 and sample[i,8]==1):      #PIR1
                    sample[i,8]=0
                    count = count+1
                if(sample[i-1,9]==0 and sample[i+1,9]==0 and sample[i,10]==0) and sample[i,9]==1:       #PIR2
                    sample[i,10]=0
                    count = count+1

    #average_std = np.zeros((8,))  
    average_std = [0,0,0,0,0,0,0,0]

    """for index in range(7):
        average_std[index] = average_persons(indexes_sensors[index],df,k_sensors_time[index]) """
  
  
    print(count)

    


    df = pd.DataFrame(sample, columns=["Date","Time","S1Temp","S2Temp","S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Persons","Overcrowded","Slope_CO2", "Parts of the day"])
    df["S1Temp"] = df["S1Temp"].diff(periods=70)
    df["S2Temp"] = df["S2Temp"].diff(periods=70)
    df["S3Temp"] = df["S3Temp"].diff(periods=70)

    return df
          
    

df = pd.read_csv("Proj1_Dataset.csv")

df["Overcrowded"] = df["Persons"]

df.loc[df["Persons"]>2, "Overcrowded"] = 1
df.loc[df["Persons"]<=2, "Overcrowded"] = 0


df_ = df.loc[df['Date'] == "11/01/2021"]
df_2=df.loc[df['Date'] == "12/01/2021"]
df_3=df.loc[df['Date'] == "13/01/2021"]
df_4=df.loc[df['Date'] == "14/01/2021"]
df_5=df.loc[df['Date'] == "15/01/2021"]
df_6=df.loc[df['Date'] == "16/01/2021"]



#Removing outliers
df_ = remove_outliers(df_)
df_2 = remove_outliers(df_2)
df_3 = remove_outliers(df_3)
df_4 = remove_outliers(df_4)
df_5 = remove_outliers(df_5)
df_6 = remove_outliers(df_6)

df_new = pd.concat([df_,df_2,df_3,df_5],axis=0)
df_new = df_new.dropna(axis=0)
df_new = df_new.drop(columns=["Time","Date"])

x = df_new.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=["S1Temp","S2Temp","S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Persons","Overcrowded","Slope_CO2", "Parts of the day"])
df = df.round(6)
df.to_csv('data_preprocessed.csv')

#Binary classification problem
X = df_new[{"S1Temp", "S2Temp", "S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Slope_CO2", "Parts of the day"}].to_numpy()
y = df_new[{"Overcrowded"}].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 

y_train= y_train.astype('int')
y_train=np.ravel(y_train)


#unique, counts = np.unique(y_train, return_counts = True)
#counter = dict(zip(unique,counts))
#print(counter)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

#unique, counts = np.unique(y_res, return_counts = True)
#counter = dict(zip(unique,counts))
#print(counter)


"""clf = MLPClassifier(solver='lbfgs',activation='relu',random_state=1, max_iter=4000,alpha=1e-6).fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train,scoring="precision", cv =10)
print("Precision: ", scores.mean())
scores = cross_val_score(clf, X_train, y_train,scoring="recall", cv =10)
print("Recall: ", scores.mean())
scores = cross_val_score(clf, X_train, y_train,scoring="accuracy", cv =10)
print("Accuracy: ", scores.mean())
scores = cross_val_score(clf, X_train, y_train,scoring="f1", cv =10)
print("F1 Score: ", scores.mean())"""



"""y_pred = clf.predict(X_train)

y_train= y_train.astype('int')
y_pred= y_pred.astype('int')

prec = precision_score(y_train,y_pred)
recall = recall_score(y_train,y_pred)
acc = accuracy_score(y_train,y_pred)
f1 = f1_score(y_train,y_pred)
print("Precision: ",prec)
print("Recall: ",recall)
print("Accuracy: ",acc)
print("F1: ",f1)"""

#Multiclass classification problem
X = df_new[{"S1Temp", "S2Temp", "S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Slope_CO2", "Parts of the day"}].to_numpy()
y = df_new[{"Persons"}].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 

y_train= y_train.astype('int')
y_train=np.ravel(y_train)
names = get_scorer_names()
#print(names)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

clf = MLPClassifier(solver='sgd',activation='relu',random_state=1, max_iter=400, learning_rate="adaptive").fit(X_res, y_res)

scoring=['precision_macro','recall_macro','accuracy','f1_macro']

scores = cross_validate(clf, X_train, y_train,scoring=scoring, cv =10)

print("Precision: ", scores["test_precision_macro"].mean())
print("Recall: ", scores["test_recall_macro"].mean())
print("F1: ", scores["test_f1_macro"].mean())
print("Accuracy: ", scores["test_accuracy"].mean())

plt.show()