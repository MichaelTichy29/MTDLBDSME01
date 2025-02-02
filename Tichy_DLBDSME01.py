import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from scipy import stats 
#
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
import csv

# import the data as csv
df_bc = pd.read_csv('Data_for_Task_1.csv', encoding = 'cp850')




#####First Impression of the data

# Get column names
column_names = df_bc.columns
print(column_names)

#shape of the original dataset
print("original bc data set")
print(df_bc.shape)
print("") 

df_test = df_bc.iloc[:,32]
print("df_test = ")
#print(df_test)

# delete an unnecessary column
df = df_bc.drop('Unnamed: 32', axis=1)
column_names = df.columns

print("bc data set")
print(df.shape)
print("") 


# Data in each calss
dfB = df.loc[df["diagnosis"] == "B"]
dfM = df.loc[df["diagnosis"] == "M"]

print("Mit M: ", dfM.shape)
print("Mit B: ", dfB.shape)


# Test for empty cells - > No empty cells.
y = df.isnull().sum()
z = df.shape
print(df.shape)
for k in range (0,z[1]): 
    print(y[k])
    print(df.columns[k])
    print("")



feature = column_names.tolist()
del feature[0]
del feature[0]



#####Visualisations ##############

#### Boxplots

# First plot all features
 
for s in feature:
    figure(num=None, figsize=(10,6))
    
    plt.subplot(1,2,1)
    sns.boxplot(x="diagnosis", y=s, data=df)
    #  
    plt.subplot(1,2,2)
    sns.violinplot(x="diagnosis", y=s, data=df)
    #
    plt.show()

#Selection of Plots - positive

figure(num=None, figsize=(10,6))
#overall height 
plt.subplot(2,2,1) 
sns.boxplot(x="diagnosis", y="radius_mean", data=df)

    
plt.subplot(2,2,2)
# separated concerning gender
sns.violinplot(x="diagnosis", y="radius_mean", data=df)


plt.subplot(2,2,3)
#ratio men
sns.boxplot(x="diagnosis", y="perimeter_mean", data=df)

plt.subplot(2,2,4)
#separated for gender and season
sns.violinplot(x="diagnosis", y="perimeter_mean", data=df)

plt.show()

#Selection of Plots - negative

figure(num=None, figsize=(10,6))
plt.subplot(2,2,1) 
sns.boxplot(x="diagnosis", y="fractal_dimension_mean", data=df)
    
plt.subplot(2,2,2)
sns.violinplot(x="diagnosis", y="fractal_dimension_mean", data=df)


plt.subplot(2,2,3)
#ratio men
sns.boxplot(x="diagnosis", y="smoothness_mean", data=df)

plt.subplot(2,2,4)
sns.violinplot(x="diagnosis", y="smoothness_mean", data=df)

plt.show()
"""

# Boxplots - old version

feature = column_names.tolist()
del feature[0]
del feature[0]


for s in feature:
  fig, ((ax1,ax2)) = plt.subplots(1,2)
  #, constraint_layout = True
  fig.tight_layout()
  # box plot
  sns.boxplot(x="diagnosis", y=s, data=df, ax=ax1)
  # violin plot
  sns.violinplot(x="diagnosis", y=s, data=df, ax=ax2)
  #
  ax1.text(-0.1, 1.1, "A", transform=ax1.transAxes,
           size=10, weight='bold')
  # numbers are position of the title "A"
  ax2.text(-0.1, 1.1, "B", transform=ax2.transAxes,
           size=10, weight='bold')
  plt.show
"""

## to compare features: two different scatterplots
""" 
scatterplot old version 
fig, (ax1, ax2) = plt.subplots(nrows = 2, sharex=False)
#positive example
sns.scatterplot(data = df, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis", ax = ax1)
#negative example
sns.scatterplot(data = df, x ="area_mean", y ="compactness_mean", palette = "bright", hue = "diagnosis", ax = ax2)
plt.show()
"""

figure(num=None, figsize=(10,6))
 
plt.subplot(1,2,1) 
sns.scatterplot(data = df, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis")    
plt.subplot(1,2,2)
sns.scatterplot(data = df, x ="area_mean", y ="compactness_mean", palette = "bright", hue = "diagnosis")
plt.show()


# t - test for the means.
# -> first measurable impression of different data
df_bcm = df.drop(df[df['diagnosis'] == "B"].index)
df_bcb = df.drop(df[df['diagnosis'] == "M"].index)

for s in feature:
    a = df_bcm[s]
    b = df_bcb[s]
    a_arr = np.array(a)
    b_arr = np.array(b)
    #ttest_ind(a_arr, b_arr)
    #equal_var = 
    print("feature is " + s)
    print(stats.ttest_ind(a_arr,b_arr))
    print("")





# logistic regression 1 Arg
# all features 
listfeature = []
listf1 = []
listacc = []
listintercept = []
listbeta = []
for s in feature:
    X = df[[s]]
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)
    
    
    reg = LogisticRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
    if f1 > 0.90:
        listfeature.append(s)
        listf1.append(round(f1,3))
        listacc.append(round(accuracy,3))
        test = reg.intercept_[0]
        listintercept.append(round(test,3))
        sertest = pd.Series(reg.coef_[:,0]) 
        #test = reg.coef_
        listbeta.append(round(sertest[0],3))
        print("feature = ")
        print(s)
        print("accuracy = ",  accuracy)
        print("f1 = ",  f1)
        print("")
        print('intercept:', reg.intercept_)
        print('coef:', reg.coef_, end='\n\n')
        print("")    
        
#list to dataframe
# export dataframe
serfeature = pd.Series(listfeature)
seracc = pd.Series(listacc)
serf1 = pd.Series(listf1)
serbeta0 = pd.Series(listintercept)
serbeta1 = pd.Series(listbeta)
dfout = pd.concat([serfeature,seracc,serf1,serbeta0, serbeta1], axis = 1)
print(dfout)
export_csv = dfout.to_csv ('export_df_log1.csv',index=False,header=True)


    
#decission rule
# perimeter worst
#df['pred'] = 
df['z'] = -(0.1802 * df['perimeter_worst'] - 19.8)
df['expz'] = np.exp(df['z'])
df['prob'] = 1/(1+df['expz'])

# Test der Zuordnung
print(df[['diagnosis','prob']]) 


plt.figure()
sns.scatterplot(data = df, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis")
plt.show()


# logistic regression 2 Arg

listfeature1 = []
listfeature2 = []
listf1 = []
listintercept = []
listbeta1 = []
listbeta2 = []

# try all combinations of features
length = len(feature)
for s in range(0,length-2):
    for t in range(s+1,length-1):
        X = df[[feature[s],feature[t]]]
        y = df['diagnosis']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)
        
        
        reg = LogisticRegression()
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
        #print("feature = ")
        #print(s, " and ", t )
        if f1 > 0.955:        
            listfeature1.append(feature[s])
            listfeature2.append(feature[t])
            listf1.append(round(f1,3))
            test = reg.intercept_[0]
            listintercept.append(round(test,3))
            sertest1 = reg.coef_[0][0]
            sertest2 = reg.coef_[0][1]
            listbeta1.append(round(sertest1,3))
            listbeta2.append(round(sertest2,3))
            print("feature = ")
            print(feature[s], feature[t])
            print("f1 = ",  f1)
            print("")
            print('intercept:', reg.intercept_)
            print('coef:', reg.coef_, end='\n\n')
            print("")    
            
        
    serfeature1 = pd.Series(listfeature1)
    serfeature2 = pd.Series(listfeature2)
    serf1 = pd.Series(listf1)
    serbeta0 = pd.Series(listintercept)
    serbeta1 = pd.Series(listbeta1)
    serbeta2 = pd.Series(listbeta2)
    
    dfout = pd.concat([serfeature1,serfeature2,serf1,serbeta0, serbeta1,serbeta2], axis = 1)
    export_csv = dfout.to_csv ('export_df_log2.csv',index=False,header=True)
        
        
#decission rule
#radius mean - perimeter worst
#df['pred'] = 
df['z'] = -(-1.16032819 * df['radius_mean'] + 0.30419355 * df['perimeter_worst'] - 16.54453422)
df['expz'] = np.exp(df['z'])
df['prob'] = 1/(1+df['expz'])


plt.figure()
sns.scatterplot(data = df, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis")
plt.show()





# For the analysis of the error.
X = df[['radius_mean','perimeter_worst']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)


reg = LogisticRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["B", "M"])
print("confusion matrix = ", confusion_matrix)


dat_pl = X_test[['radius_mean','perimeter_worst']]
dat_pl['diagnosis'] = y_test
dat_pl1 = X_test[['radius_mean','perimeter_worst']]
dat_pl1['diagnosis'] = y_pred

figure(num=None, figsize=(10,6))
#overall height 
plt.subplot(1,2,1) 
sns.scatterplot(data = dat_pl, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis")
plt.title("Truth classes", loc= 'left', weight='bold', size=12)
    
plt.subplot(1,2,2)
# separated concerning gender
sns.scatterplot(data = dat_pl1, x ="radius_mean", y ="perimeter_worst", palette = "bright", hue = "diagnosis")
plt.title("Predicted classes", loc= 'left', weight='bold', size=12)



#To see the possible steps: Logistig regression with 3 Arguments

length = len(feature)
for s in range(0,length-3):
    for t in range(s+1,length-2):
        for u in range(t+1,length-1):
        
            X = df[[feature[s],feature[t],feature[u]]]
            y = df['diagnosis']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)
            
            
            reg = LogisticRegression()
            reg.fit(X_train,y_train)
            y_pred = reg.predict(X_test)
            
            accuracy = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
            #print("feature = ")
            #print(s, " and ", t )
            if f1 > 0.97:
                print(feature[s], ",",feature[t], " and ", feature[u] )
                #print("accuracy = ",  accuracy)
                print("f1 = ",  f1)
                print("")



################ New Model
# simple trees - > Not that high f1 scores
X = df.drop('id', axis=1)
X = X.drop('diagnosis', axis=1)

y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)

decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=2)
#decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=2)

decision_tree = decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
print("acc = ", accuracy)
print("f1 = ", f1)

# For the analysis of the error.
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["B", "M"])
print("confusion matrix = ", confusion_matrix)

y_name = np.unique(y)
print("y_name = ", y_name)
frequency_counts = y_train.value_counts()
print(frequency_counts)


# To visualize the tree
sklearn.tree.plot_tree(decision_tree, feature_names=feature, class_names=y_name, filled=True, fontsize=8)
                       #, *, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)



# LDA with two features

listfeature1 = []
listfeature2 = []
listf1 = []
listacc = []

X = df.drop('id', axis=1)
X = X.drop('diagnosis', axis=1)
y = df['diagnosis']

length = len(feature)
for s in range(0,length-2):
    for t in range(s+1,length-1):    
        Xn = X[[feature[s],feature[t]]] 
        
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, random_state=24, train_size = 0.80)
        
        LDA = LinearDiscriminantAnalysis()
        data_projected = LDA.fit_transform(X_train,y_train)
        
        y_pred = LDA.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
        if f1 > 0.96:
            print("features: ", feature[s], feature[t])
            print("acc = ", accuracy)
            print("f1 = ", f1)
            print("")
            listfeature1.append(feature[s])
            listfeature2.append(feature[t])
            listf1.append(round(f1,3))
            listacc.append(round(accuracy,3))
            #
serfeature1 = pd.Series(listfeature1)
serfeature2 = pd.Series(listfeature2)
serf1 = pd.Series(listf1)
seracc = pd.Series(listacc)

dfout = pd.concat([serfeature1,serfeature2,serf1,seracc], axis = 1)
export_csv = dfout.to_csv ('export_df_lda.csv',index=False,header=True)


# LDA: Linear discriminanz analysis - > for two selected features.
# With analysis of error
X = df.drop('id', axis=1)
#y = df['diagnosis']

X = X[['diagnosis','perimeter_worst', 'smoothness_worst']]
y = X['diagnosis']
X_train1, X_test1, y_train, y_test = train_test_split(X, y, random_state=24, train_size = 0.80)

#X_train1, X_test1 = train_test_split(X, random_state=24, train_size = 0.80)


X_train = X_train1[['perimeter_worst', 'smoothness_worst']]
X_test = X_test1[['perimeter_worst', 'smoothness_worst']]
#y_train = X_train1[['diagnosis']]
#y_test = X_test1[['diagnosis']]


LDA = LinearDiscriminantAnalysis()
data_projected = LDA.fit_transform(X_train,y_train)

y_pred = LDA.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
print("LDA")
print("acc = ", accuracy)
print("f1 = ", f1)

#confusion matrix for the analyis of error
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["B", "M"])
print("confusion matrix = ", confusion_matrix)


# convert the data for representation
print(type(y_pred))
print(type(y_test))
 
dfy1 = pd.DataFrame({"diag":[y_pred]})
serdf = dfy1['diag']
dfy2 = pd.concat([serdf], axis = 1)
frequency_counts = dfy2["diag"].value_counts()
print("Pred")
print(frequency_counts)


dfy = pd.concat([y_test,serdf], axis = 1)
y_name = np.unique(y_test)
column_names = dfy.columns

frequency_counts = dfy["diagnosis"].value_counts()
print("TEST")
print(frequency_counts)


#construct the seperating line for the LDA
# for the means
X_muB = X_train1.loc[df["diagnosis"] == "B"]
X_muM = X_train1.loc[df["diagnosis"] == "M"]
X_muB = X_muB[['perimeter_worst', 'smoothness_worst']]
X_muM = X_muM[['perimeter_worst', 'smoothness_worst']]
mu_M = X_muM.mean()
mu_B = X_muB.mean()

ser_per = X_train['perimeter_worst']
ser_smo = X_train['smoothness_worst']

dat = np.array([ser_per, ser_smo])

cov = np.cov(dat, bias=True)
cov_inv = np.linalg.inv(cov)

print("mu_M = ", mu_M)
print("mu_B = ", mu_B)

print("cov = ", cov)
print("cov_inv = ", cov_inv)
vt = np.mat(cov_inv) * np.transpose(np.mat(mu_M) - np.mat(mu_B))
w = 0.5*( np.mat(mu_M) + np.mat(mu_B)) 
# How to make a decision
print("w = ", w)
print("vt = ", vt)
print("((x,y) - w) * vt >0 => nach M, sonst B")
  


# To see the possible steps:
# LDA with four features
X = df.drop('id', axis=1)
X = X.drop('diagnosis', axis=1)
y = df['diagnosis']
f1max = 0.8
length = len(feature)
for s in range(0,length-3):
    for t in range(s+1,length-2):
        for z in range(t+1,length-1):
            for v in range(z+1,length):
                Xn = X[[feature[s],feature[t], feature[z], feature[v]]] 
                
                X_train, X_test, y_train, y_test = train_test_split(Xn, y, random_state=24, train_size = 0.80)
                
                LDA = LinearDiscriminantAnalysis()
                data_projected = LDA.fit_transform(X_train,y_train)
                
                y_pred = LDA.predict(X_test)
                
                accuracy = metrics.accuracy_score(y_test, y_pred)
                f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label="B", average='binary')
                if f1 > f1max:
                    print("features: ", feature[s], feature[t], feature[z], feature[v])
                    print("acc = ", accuracy)
                    print("f1 = ", f1)
                    print("")
                    f1max = f1

