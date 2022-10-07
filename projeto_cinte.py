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

def function_plot(frame):
    #frame.plot(x='Time', y="S1Temp", title='Date = 11/01/2021')  
    #frame.plot(x='Time', y="S2Temp", title='Date = 11/01/2021')  
    #frame.plot(x='Time', y="S3Temp", title='Date = 11/01/2021')  
    #frame.plot(x='Time', y="S1Light", title='Date = 11/01/2021')  
    #frame.plot(x='Time', y="S2Light", title='Date = 11/01/2021')    
    #frame.plot(x='Time', y="S3Light", title='Date = 11/01/2021')    
    #frame.plot(x='Time', y="CO2")
    #frame.plot(x='Time', y="PIR1")
    frame.plot(x='Time', y="PIR2")
    frame.plot(x='Time', y="Persons")

    

def remove_outliers(df):
    average_s1temp = df["S1Temp"].rolling(window=5).mean()
    std_s1temp = df["S1Temp"].rolling(window=5).std()

    average_s2temp = df["S2Temp"].rolling(window=5).mean()
    std_s2temp = df["S2Temp"].rolling(window=5).std()

    average_s3temp = df["S3Temp"].rolling(window=5).mean()
    std_s3temp = df["S3Temp"].rolling(window=5).std()

    average_s1light = df["S1Light"].rolling(window=8).mean()
    std_s1light = df["S1Light"].rolling(window=8).std()

    average_s2light = df["S2Light"].rolling(window=5).mean()
    std_s2light = df["S2Light"].rolling(window=5).std()

    average_s3light = df["S3Light"].rolling(window=5).mean()
    std_s3light = df["S3Light"].rolling(window=5).std()

    average_co2 = df["CO2"].rolling(window=5).mean()
    std_co2 = df["CO2"].rolling(window=5).std()

    average_pir1 = df["PIR1"].rolling(window=20).mean()
    std_pir1 = df["PIR1"].rolling(window=20).std()

    average_pir2 = df["PIR2"].rolling(window=20).mean()
    std_pir2 = df["PIR2"].rolling(window=20).std()

    sample = pd.DataFrame(df).to_numpy()


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
            if(sample[i,2] > average_s1temp[i]+1.75*std_s1temp[i] or sample[i,2] < average_s1temp[i]-1.75*std_s1temp[i]):
                sample[i,2]=sample[i-1,2]

            if(sample[i,3] > average_s2temp[i]+1.8*std_s2temp[i] or sample[i,3] < average_s2temp[i]-1.8*std_s2temp[i]):
                sample[i,3]=sample[i-1,3]

            if(sample[i,4] > average_s3temp[i]+1.75*std_s3temp[i] or sample[i,4] < average_s3temp[i]-1.75*std_s3temp[i]):
                sample[i,4]=sample[i-1,4]

            if(sample[i,5] > average_s1light[i]+2.45*std_s1light[i] or sample[i,5] < average_s1light[i]-2.45*std_s1light[i]):
                sample[i,5]=sample[i-1,5]
            elif(sample[i,5] > 1000):       #Especific case: Date 12/01
                sample[i,5]=sample[i-1,5]

            if(sample[i,6] > average_s2light[i]+1.8*std_s2light[i] or sample[i,6] < average_s2light[i]-1.8*std_s2light[i]):
                sample[i,6]=sample[i-1,6]

            if(sample[i,7] > average_s3light[i]+1.78*std_s3light[i] or sample[i,7] < average_s3light[i]-1.78*std_s3light[i]):
                sample[i,7]=sample[i-1,7]

            if(sample[i,8] > average_co2[i]+1.8*std_co2[i] or sample[i,8] < average_co2[i]-1.8*std_co2[i]):
                sample[i,8]=sample[i-1,8]

            if((sample[i,9] > average_pir1[i]+2*std_pir1[i] or sample[i,9] < average_pir1[i]-2*std_pir1[i]) and sample[i,11] == 0):
                sample[i,9]=sample[i-1,9]

            if((sample[i,10] > average_pir2[i]+2*std_pir2[i] or sample[i,10] < average_pir2[i]-2*std_pir2[i]) and sample[i,11]==0):
                sample[i,10]=sample[i-1,10]

        if(i>=1 and i < len(df)-1):
         
            if(sample[i-1,9]==0 and sample[i+1,9]==0 and sample[i,11]==0):      #PIR1
                sample[i,9]=0
            if(sample[i-1,10]==0 and sample[i+1,10]==0 and sample[i,11]==0):       #PIR2
                sample[i,11]=0
    
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

function_plot(df_2)

#Removing outliers
df = remove_outliers(df)
    
df_ = df.loc[df['Date'] == "11/01/2021"]
df_2 = df.loc[df['Date'] == "12/01/2021"]
df_3 = df.loc[df['Date'] == "13/01/2021"]
df_4 = df.loc[df['Date'] == "14/01/2021"]
df_5 = df.loc[df['Date'] == "15/01/2021"]
df_6 = df.loc[df['Date'] == "16/01/2021"]

function_plot(df_2)

plt.show()