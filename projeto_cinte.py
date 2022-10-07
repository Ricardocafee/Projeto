import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d, Axes3D

indexes_sensors = {
    0:"S1Temp",
    1:"S2Temp",
    2:"S3Temp",
    3:"S1Light",
    4:"S2Light",
    5:"S3Light",
    6:"CO2",
    7:"PIR1",
    8:"PIR2",
    9:"Persons",
}

window_sensors = {
    0:5,         #S1Temp
    1:5,         #S2Temp
    2:5,         #S3Temp
    3:10,         #S1Light
    4:5,         #S2Light
    5:5,         #S3Light
    6:5,         #CO2
    7: 20,       #PIR1
    8: 20,       #PIR2
}

k_sensors = {
    0:1.75,     #S1Temp
    1:1.8,      #S2Temp
    2:1.75,     #S3Temp
    3:2.7,     #S1Light
    4:1.8,      #S2Light
    5:1.8,     #S3Light
    6:1.8,      #CO2
    7: 2,       #PIR1
    8: 2,       #PIR2
}

k_sensors_time = {
    0:5,     #S1Temp
    1:6,      #S2Temp
    2:5,     #S3Temp
    3:6,     #S1Light
    4:6,      #S2Light
    5:6,     #S3Light
    6:1000,      #CO2
    7: 1000,       #PIR1
    8: 1000,       #PIR2
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

    #for index in range(10):
       # frame.plot(x="Time",y=indexes_sensors[index])

    #plt.scatter(x=frame["CO2"], y = frame["Persons"])
    #plt.xlim([0,400])
    frame.plot(x="Time", y = "S3Light")

def remove_outliers(df):

    #Based on time


    average_time = np.zeros((8,len(df)))
    std_time = np.zeros((8,len(df)))  

    #Based on time

    for index in range(8):
        average_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).mean()
        std_time[index]=df[indexes_sensors[index]].rolling(window=window_sensors[index]).std()

    sample = pd.DataFrame(df).to_numpy()
        
    count = 0     

    average_std = [0,0,0,0,0,0,0,0]

    for i in range(len(df)):
        time = sample[i,1]
        time = int(time[:2])

        #Out of work time
        if(time<6 or time > 21):
            sample[i,5] = 0     #S1Light
            sample[i,6] = 0     #S2Light
            sample[i,7] = 0     #S3Light
            sample[i,9] = 0     #PIR1
            sample[i,10] = 0     #PIR2

        if(i>4):
            for j in range(8):

                if(sample[i,j+2] > average_time[j][i]+k_sensors[j]*std_time[j][i] or sample[i,j+2] < average_time[j][i] - k_sensors[j]*std_time[j][i]):
                    sample[i,j+2]=sample[i-1,j+2]
                    #print(indexes_sensors[j])
                    count = count+1

                
            if(sample[i,5] > 1000):       #Especific case: Date 12/01
                sample[i,5]=sample[i-1,5]  #S1Light
                #count = count+1

            if(i>=1 and i < len(df)-1):
            
                if(sample[i-1,9]==0 and sample[i+1,9]==0 and sample[i,11]==0 and sample[i,9]==1):      #PIR1
                    sample[i,9]=0
                    count = count+1
                if(sample[i-1,10]==0 and sample[i+1,10]==0 and sample[i,11]==0) and sample[i,10]==1:       #PIR2
                    sample[i,11]=0
                    count = count+1

    #average_std = np.zeros((8,))  
    average_std = [0,0,0,0,0,0,0,0]

    for index in range(8):
        average_std[index] = average_persons(indexes_sensors[index],df,k_sensors_time[index]) 

    
    for i in range(len(df)):
        
    
        for j in range(6):

            if(sample[i,11] == 0):
                if(sample[i,j+2] > average_std[j].mean0+average_std[j].k*average_std[j].std0 or sample[i,j+2] < average_std[j].mean0-average_std[j].k*average_std[j].std0):
                    sample[i,j+2]=average_std[j].mean0
                    
                    count=count+1
            if(sample[i,11] == 1):
                if(sample[i,j+2] > average_std[j].mean1+average_std[j].k*average_std[j].std1 or sample[i,j+2] < average_std[j].mean1-average_std[j].k*average_std[j].std1):
                    sample[i,j+2]=average_std[j].mean1
                    
                    count=count+1
            if(sample[i,11] == 2):
                if(sample[i,j+2] > average_std[j].mean2+average_std[j].k*average_std[j].std2 or sample[i,j+2] < average_std[j].mean2-average_std[j].k*average_std[j].std2):
                    sample[i,j+2]=average_std[j].mean2
                    
                    count=count+1
            if(sample[i,11] == 3):
                if(sample[i,j+2] > average_std[j].mean3+average_std[j].k*average_std[j].std3 or sample[i,j+2] < average_std[j].mean3-average_std[j].k*average_std[j].std3):
                    sample[i,j+2]=average_std[j].mean3
                    
                    count=count+1
    print(count)
                        

                
            
        
    df = pd.DataFrame(sample, columns=["Date","Time","S1Temp","S2Temp","S3Temp","S1Light","S2Light","S3Light","CO2","PIR1","PIR2","Persons","Overcrowded"])
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

df_new = pd.concat([df_,df_2,df_3,df_4,df_5,df_6],axis=0)



plt.show()