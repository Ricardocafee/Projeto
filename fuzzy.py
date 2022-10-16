import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, make_scorer, get_scorer_names

names =['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','PIR1','PIR2','Persons','Overcrowded','Slope_CO2','Parts of the day']

df = pd.read_csv('data_preprocessed.csv',usecols=names)



df['AvTemp'] = 100*(df['S1Temp'] + df['S2Temp'])/2
df['AvLight'] = (df['S1Light'] + df['S2Light'] + df['S3Light'])/3



df['2nd_slope'] = df['Slope_CO2'].diff(periods=50)
df = df.round()
df['AvPIR'] = (df['PIR1'] + df['PIR2'])/2
df['AvPIR'] = df['PIR1']
#print(df)
#print(df['Parts of the day'].max())

#input

X = df[{'AvTemp','AvLight','2nd_slope','Slope_CO2'}].to_numpy()
y = df[{"Overcrowded"}].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 

y_train= y_train.astype('int')
y_train=np.ravel(y_train)

df_x = pd.DataFrame(X_train, columns=["AvTemp","AvLight","Slope_CO2","2nd_slope"])
df_y = pd.DataFrame(y_train, columns=["Overcrowded"])

df_x_test = pd.DataFrame(X_test, columns=["AvTemp","AvLight","Slope_CO2","2nd_slope"])
df_y_test = pd.DataFrame(y_test, columns=["Overcrowded"])

df = pd.concat([df_x,df_y], axis = 1, join = 'inner')
df_new = pd.concat([df_x_test,df_y_test], axis = 1, join = 'inner')

light = ctrl.Antecedent(np.arange(0, 501, 1), 'light')
pir = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'pir')
c02_slope = ctrl.Antecedent(np.arange(-300, 301, 1), 'c02_slope')
c02_slope2nd = ctrl.Antecedent(np.arange(-300, 301, 1), 'c02_slope2nd')
temp = ctrl.Antecedent(np.arange(-100, 101, 1), 'temp')
#output
overcrowded = ctrl.Consequent(np.arange(0, 2, 1), 'overcrowded')


#input membership
temp['low'] = fuzz.trapmf(temp.universe, [-100, -100, -40,-20])
temp['medium'] = fuzz.trapmf(temp.universe, [-30, -20,20, 30])
temp['high'] = fuzz.trapmf(temp.universe, [20, 30, 100,100])


light['low'] = fuzz.trapmf(light.universe, [0, 0, 90,110])
light['medium_low'] = fuzz.trapmf(light.universe, [90, 110, 190,210])
light['medium_high'] = fuzz.trapmf(light.universe, [190, 210, 340,360])
light['high'] = fuzz.trapmf(light.universe, [340, 360, 500,500])

c02_slope['negative'] = fuzz.trimf(c02_slope.universe, [-300, -300, 25])
c02_slope['positive'] = fuzz.trimf(c02_slope.universe, [-25, 300, 300])

c02_slope2nd['negative'] = fuzz.trapmf(c02_slope.universe, [-300, -300,-50, 0])
c02_slope2nd['constante'] = fuzz.trapmf(c02_slope.universe, [-25, 0, 25,50])
c02_slope2nd['positive'] = fuzz.trapmf(c02_slope.universe, [25, 75, 300,300])

pir['low'] = fuzz.trimf(pir.universe, [0, 0, 0.3])
pir['middle'] = fuzz.trimf(pir.universe, [0.2, 0.5, 0.8])
pir['high'] = fuzz.trimf(pir.universe, [0.7, 1, 1])

#output membership
overcrowded['false'] = fuzz.trimf(overcrowded.universe, [0, 0, 1])
overcrowded['true'] = fuzz.trimf(overcrowded.universe, [0, 1, 1])

#view membership
""""
temp.view()
light.view()
pir.view()
c02_slope.view()
c02_slope2nd.view()
#overcrowded.view()
plt.show()
"""
############# rules
"""
rule1 = ctrl.Rule(light['high']        & (c02_slope['negative'] | pir['high']),overcrowded['false'])
rule2 = ctrl.Rule(light['medium_high'] & (c02_slope['positive'] | pir['high']),overcrowded['true'])

rule1 = ctrl.Rule(light['high']        & (c02_slope['positive'] | pir['low']) ,overcrowded['false'])
rule2 = ctrl.Rule(light['medium_high'] & (c02_slope['negative'] | pir['low']) ,overcrowded['false'])

rule3 = ctrl.Rule(light['low']                                                ,overcrowded['false'])
rule4 = ctrl.Rule(light['medium_low']                                         ,overcrowded['false'])

rule5 = ctrl.Rule(light['medium_high']                                        ,overcrowded['false'])
rule6 = ctrl.Rule(light['high']                                               ,overcrowded['true'])
"""

#rule1.view()
#plt.show()

"""rule1 = ctrl.Rule(light['high'] ,overcrowded['true'])
rule2 = ctrl.Rule((light['medium_high'] | light['medium_low'] | light['low']), overcrowded['false'])
"""



rule1 = ctrl.Rule(c02_slope['positive'] & c02_slope2nd['positive'] & temp['high'],overcrowded['true'])
rule2 = ctrl.Rule(light['medium_low'] & ~(c02_slope['positive'] & c02_slope2nd['positive'] & temp['high']),overcrowded['false'])
rule3 = ctrl.Rule((light['medium_high']|light['high']) & c02_slope['positive'] & c02_slope2nd['positive'] & temp['medium'],overcrowded['true'])
rule4 = ctrl.Rule((light['medium_high'] & ~(c02_slope['positive'] & c02_slope2nd['positive'] & (temp['medium']|temp['high']))),overcrowded['false'])
rule5 = ctrl.Rule((light['high'] & temp['high']),overcrowded['true'])
rule6 = ctrl.Rule((light['high'] & temp['low']) ,overcrowded['false'])
rule7 = ctrl.Rule((light['high'] & c02_slope['negative'] & (c02_slope2nd['positive'] | c02_slope2nd['negative'])& temp['medium']),overcrowded['false'])
rule8 = ctrl.Rule((light['high'] & (c02_slope['negative'] | c02_slope['positive']) & (c02_slope2nd['constante'] | c02_slope2nd['negative']) & temp['medium']),overcrowded['true'])
rule9 = ctrl.Rule((light['low'] & c02_slope['positive'] & c02_slope2nd['positive'] & temp['medium']),overcrowded['true'])
rule10 = ctrl.Rule((light['low'] & (c02_slope['negative'] | c02_slope2nd['negative'])),overcrowded['false'])
rule11 = ctrl.Rule((light['low'] & c02_slope['positive'] & (c02_slope2nd['constante'] | c02_slope2nd['positive']) & temp['low']),overcrowded['false'])
rule12 = ctrl.Rule((light['low'] & c02_slope['positive'] & c02_slope2nd['constante'] & temp['high']),overcrowded['false'])
rule13 = ctrl.Rule((light['low'] & c02_slope['positive'] & c02_slope2nd['constante'] & temp['medium']),overcrowded['false'])

"""rule1 = ctrl.Rule(light['high'],overcrowded['true'])
rule2 = ctrl.Rule(light['medium_high'],overcrowded['false'])
rule3 = ctrl.Rule(light['medium_low'],overcrowded['false'])
rule4 = ctrl.Rule(light['low'] & (c02_slope2nd['constante'] | c02_slope2nd['negative']),overcrowded['false'])
rule5 = ctrl.Rule(light['low'] & c02_slope2nd['positive'] & pir['high'],overcrowded['true'])
rule6 = ctrl.Rule(light['low'] & c02_slope2nd['positive'] & pir['low'],overcrowded['false'])
rule7 = ctrl.Rule(light['low'] & c02_slope2nd['positive'] & pir['middle'],overcrowded['true'])
rule8 = ctrl.Rule(temp['high'],overcrowded['true'])
rule9 = ctrl.Rule(c02_slope['positive'] & temp['low'],overcrowded['false'])
"""


overcrowded_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13])
#overcrowded_ctrl = ctrl.ControlSystem([rule1, rule2])

over = ctrl.ControlSystemSimulation(overcrowded_ctrl)

df['est_overcrowded'] = np.nan

for index,row in df.iterrows():
    #if(row)
    over.input['light'] = row['AvLight']
    over.input['c02_slope2nd'] = row['2nd_slope']
    over.input['temp'] = row['AvTemp']
    over.input['c02_slope'] = row['Slope_CO2']
    #Crunch the numbers
    over.compute()
    df.iloc[index,5] = over.output['overcrowded'].round()

#print(df)

colors = ['blue','red']
colormap = matplotlib.colors.ListedColormap(colors)
x = df.index

plt.figure()
plt.title("Light")
y = df['AvLight']
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)

plt.figure()
plt.title("CO2")
y = df['Slope_CO2']
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)


"""plt.figure()
plt.title("Persons")
y = df['Persons']
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)"""

plt.figure()
plt.title("co2 declive")
y = df['Slope_CO2'].diff(periods=50)
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)

plt.figure()
plt.title("temp")
y = df['AvTemp']
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)

"""plt.figure()
plt.title("pir")
y = df['AvPIR']
plt. scatter(x, y, c=np.where(df['Overcrowded'] == df['est_overcrowded'], 0, 1), cmap=colormap)"""

#check TP fp fn tn
df['tp'] = np.where((df['Overcrowded'] == 1) & (df['est_overcrowded'] == 1), 1, 0)
df['fp'] = np.where((df['Overcrowded'] == 0) & (df['est_overcrowded'] == 1), 1, 0)
df['fn'] = np.where((df['Overcrowded'] == 1) & (df['est_overcrowded'] == 0), 1, 0)
df['tn'] = np.where((df['Overcrowded'] == 0) & (df['est_overcrowded'] == 0), 1, 0)

tp = df['tp'].sum()
fp = df['fp'].sum()
fn = df['fn'].sum()
tn = df['tn'].sum()

accuracy = (tp+tn)/(tp+fp+fn+tn)
precision = tp/(tp+fp)   
recall = tp/(tp+fn)
specificity = tn/(tn+fp)

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"specificity: {specificity}")

##Classifier Binary

x = df[["AvTemp","AvLight","Slope_CO2","2nd_slope","Overcrowded"]].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_new = pd.DataFrame(x_scaled, columns=["AvTemp","AvLight","Slope_CO2","2nd_slope","Overcrowded"])
df_new = df_new.dropna(axis=0)
X = df_new[{"AvTemp","AvLight","Slope_CO2","2nd_slope"}].to_numpy()
y = df_new[{"Overcrowded"}].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) 

y_train= y_train.astype('int')
y_train=np.ravel(y_train)


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

clf = MLPClassifier(solver='lbfgs',activation='relu',random_state=1, max_iter=4000).fit(X_res, y_res)

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
