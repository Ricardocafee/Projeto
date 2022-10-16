from typing import Counter
import numpy as np
import pandas as pd
import joblib
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
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
}

k_sensors = {
    0:1.75,     #S1Temp
    1:2,      #S2Temp
    2:1.75,     #S3Temp
    3:1.78,     #S1Light
    4:1.78,      #S2Light
    5:1.78,     #S3Light
}

def remove_outliers(df):

    #Removal of CO2 and replacement with diff CO2
    df_new = df["CO2"].diff(periods=50)
    df = df.assign(Slope_CO2=df_new)
    df = df.drop(columns=["CO2"])
    

    #Based on time
    average_time = np.zeros((7,len(df)))
    std_time = np.zeros((7,len(df))) 

    for index in range(5):
        average_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).mean()
        std_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).std()


    #Convert to Numpy
    sample = pd.DataFrame(df).to_numpy()
    z = np.zeros((len(df),1), dtype=object)
    sample = np.append(sample,z,axis=1)

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



    #Outlier removal based on the past only
    for i in range(len(df)):
        if(i>0):
            for j in range(5):
                if(sample[i,j+2] > average_time[j][i]+k_sensors[j]*std_time[j][i] or sample[i,j+2] < average_time[j][i] - k_sensors[j]*std_time[j][i]):
                            sample[i,j+2]=sample[i-1,j+2]



    df = pd.DataFrame(sample, columns=["Date","Time","S1Temp","S2Temp","S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Persons","Overcrowded","Slope_CO2", "Parts of the day"]) 
    df["S1Temp"] = df["S1Temp"].diff(periods=50)
    df["S2Temp"] = df["S2Temp"].diff(periods=50)
    df["S3Temp"] = df["S3Temp"].diff(periods=50)
    
    
    return df
                            

filename = "finalized_model.sav"
loaded_model = joblib.load(filename)
try:
    filename_test = str(sys.argv[1])
except:
    print("Input on the command line missing or wrong")
    sys.exit(1)

filename_termination = ".csv"
filename_test=filename_test+filename_termination

try:
    df = pd.read_csv(filename_test)
except FileNotFoundError:
    print("File not found.")
except pd.errors.EmptyDataError:
    print("No data.")
except pd.errors.ParserError:
    print("ParseError")
except Exception:
    print("Some other exception.")

df["Overcrowded"] = df["Persons"]
df.loc[df["Persons"]>2, "Overcrowded"] = 1
df.loc[df["Persons"]<=2, "Overcrowded"] = 0

df = remove_outliers(df)
df = df.drop(columns=["Time","Date"])


df = df.drop(columns=["S2Temp"])
df = df.fillna(0)

X_test = df[["S1Temp", "S3Temp","S1Light","S2Light","S3Light","PIR1","PIR2","Slope_CO2", "Parts of the day"]].to_numpy()
y_test = df[["Persons"]].to_numpy()
y_pred = loaded_model.predict(X_test)

lab = preprocessing.LabelEncoder()
y_transformer = lab.fit_transform(y_pred)
lab_test = preprocessing.LabelEncoder()
y_test_trans = lab.fit_transform(y_test)


print("#################################################")
print("Multi-Class Classification Problem - Test Set Given by Prof")
print("#################################################\n")
prec = precision_score(y_test_trans,y_transformer,average='macro')
recall = recall_score(y_test_trans,y_transformer, average='macro')
acc = accuracy_score(y_test_trans,y_transformer)
f1 = f1_score(y_test_trans,y_transformer, average='macro')
print("Precision macro: ",prec)
print("Recall macro: ",recall)
print("Accuracy: ",acc)
print("F1 macro: ",f1)

confusion_mat = confusion_matrix(y_test_trans,y_transformer) 
prec_class = precision_score(y_test_trans,y_transformer, average=None)
recall_class = recall_score(y_test_trans,y_transformer, average=None)
print("Precision class: ",prec_class)
print("Recall class: ",recall_class)
print("Confusion matrix: ",confusion_mat)
