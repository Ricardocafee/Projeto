import numpy as np
import pandas as pd
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

df = pd.read_csv('Proj1_Dataset.csv')
print(df)

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