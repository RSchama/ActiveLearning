#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:58:19 2022

@author: schama
"""

#libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt


##################################################
#Random Active Learning function
def RandomActiveLearning(model,xtrain,xtest,ytrain,ytest,unl,N=50,k=10):
    """
    Active learning random sampling algorithm.
    Parameters
    ----------
    model : classifier
        A classification algorithm that can handle multiple classes. 
        Scikit-Learn algorithms that have 'fit', 'predict', will work well.
    xtrain : dataframe
        Training Matrix.
    xtest : dataframe
        Testing Matrix.
    ytrain : 1D array
        Training Labels.
    ytest : 1D array
        Testing Labels.
    unl : dataframe
        Matrix with unlabeled data and last column of Labels named 'label'.
    N : integer, optional
        Number of iterations. The default is 50.
    k : integer, optional
        Number of samples to get from unlabeled at each iteration. The default is 10.

    Returns
    -------
    MiscTrain (list of accuracy for train data at each iteration),
    MiscTest (list of accuracy for test data at each iteration),
    DataSize (List with train data size at each iteration),
    iterations (List with iteration number).

    """
    
    MiscTrain = []
    MiscTest = []
    DataSize = []
    iterations = []
    
    for i in range(N):
        model.fit(xtrain, ytrain)
        predtrain = model.predict(xtrain)
        acctrain = accuracy_score(ytrain, predtrain)
        predtest = model.predict(xtest)
        acctest = accuracy_score(ytest, predtest)
        MiscTrain.append(acctrain)
        MiscTest.append(acctest)  
        
        sample = unl.sample(n=k, replace = False)      
        chosen_idx = sample.index
        x = sample.loc[:, sample.columns != 'label']
        xtrain = pd.concat([xtrain, x], ignore_index = True)
        
        ytrain = np.append(ytrain, sample['label'])
        
        unl.drop(index=chosen_idx, inplace=True)  
        unl.reset_index(drop=True)
        
        DataSize.append(xtrain.shape[0])
        iterations.append(i+1)
        print(unl.shape, xtrain.shape)
        
    return(MiscTrain,MiscTest,DataSize,iterations)
##################################################


##################################################
#the function for entropy-based active learning
def EntropyActiveLearning(model,xtrain,xtest,ytrain,ytest,unl,N=50,k=10):
    """
    Active learning entropy-based algorithm.
    Parameters
    ----------
    model : classifier
        A classification algorithm that can handle multiple classes. 
        Scikit-Learn algorithms that have 'fit', 'predict' and 'predict_proba', will work well.
    xtrain : dataframe
        Training Matrix.
    xtest : dataframe
        Testing Matrix.
    ytrain : 1D array
        Training Labels.
    ytest : 1D array
        Testing Labels.
    unl : dataframe
        Matrix with unlabeled data and last column of Labels named 'label'.
    N : integer, optional
        Number of iterations. The default is 50.
    k : integer, optional
        Number of samples to get from unlabeled at each iteration. The default is 10.

    Returns
    -------
    MiscTrain (list of accuracy for train data at each iteration),
    MiscTest (list of accuracy for test data at each iteration),
    DataSize (List with train data size at each iteration),
    iterations (List with iteration number).

    """
    
    MiscTrain = []
    MiscTest = []
    DataSize = []
    iterations = []
    
    for i in range(N):
        model.fit(xtrain, ytrain)
        predtrain = model.predict(xtrain)
        acctrain = accuracy_score(ytrain, predtrain)
        predtest = model.predict(xtest)
        acctest = accuracy_score(ytest, predtest)
        MiscTrain.append(acctrain)
        MiscTest.append(acctest)  
        
        prob = model.predict_proba(unl.loc[:,0:99])
        unl['entropy'] = (-prob*np.log2(prob)).sum(axis=1)
        
        # sort the DataFrame (entropy bigger is more uncertain, so sort descending)
        unl = unl.sort_values('entropy', ascending=False, ignore_index=True)
        # get the top ten ones
        sample = unl.iloc[0:k,:]      
        chosen_idx = sample.index
        x = sample.loc[:, 0:99]
        
        xtrain = pd.concat([xtrain, x], ignore_index = True)
        
        ytrain = np.append(ytrain, sample['label'])
        
        unl.drop(index=chosen_idx, inplace=True)  
        unl.reset_index(drop=True)
        
        DataSize.append(xtrain.shape[0])
        iterations.append(i+1)
        print(unl.shape, xtrain.shape)
        
    return(MiscTrain,MiscTest,DataSize,iterations)
##################################################

##################################################
#function for importing several files in a folder with .mat format (Matlab)
#saves each file data as a global variabel and return the names of these variables

def importMatLab(path):
    """
    Function for importing several files in a folder with .mat format (Matlab).
    Saves each file data as a global variable and returns the names of these variables.
    The data is saved as a dataframe.
    
    Parameters
    ----------
    path : string
        the path to the folder where the .mat files are (only them should be in the file).
    Returns
    -------
    names (list of names of the variables that were created),

    """
    
    #getting the file names
    Mindlist = [f for f in listdir(path) if isfile(join(path, f))]
    #list of filenames in folder
    
    #getting the number of files in the folder
    n=len(Mindlist)
    
    #just making a copy to fill in with the different mat dictionaries    
    Mindfiles = Mindlist.copy() 
    #reading each file in the directory and saving as an item in the list
    for i in range(n):
        Mindfiles[i] = loadmat(join(path, Mindlist[i]))
    #This is a list of dictionaries with for key:value items inside, the matrix/label is the last item in each
    #Mindfiles[1]
    
    #just creating a placeholder to fill wiith the last item od the dictionaries
    MindMatrixlist = Mindlist.copy()
    #iterating over the dictionary list and extracting the last item to a new list
    for i in range(n):
        MindMatrixlist[i] = [x for x in Mindfiles[i].popitem()]
    #the bolw list contains 18 lists with two elements each: the name of the data set and its numpy array
    #MindMatrixlist[0]
    
    #saving the name of the files in the order they were imported
    names1 = Mindlist.copy()
    #getting rid of  the .mat in the end
    # using list comprehension + list slicing
    # remove last character from list of strings
    names = [nm[ : -4] for nm in names1]
    
    #generating global variabels with the names above, each with the second element of the sublist
    for i in range(n):
        globals()[str(names[i])] = pd.DataFrame(MindMatrixlist[i][1])
    return(names)
##################################################

#MndReading data
#saving the path to the folder where the files are
pathMind = "/Users/schama/Desktop/Data/MindReading/"
#calling the function to import the data
names = importMatLab(pathMind)

#saving variables to new names and flattening the labels
ytest_1 = np.ravel(testingLabels_MindReading1.copy())
ytest_2 = np.ravel(testingLabels_MindReading2.copy())
ytest_3 = np.ravel(testingLabels_MindReading3.copy())
xtest_1 = testingMatrix_MindReading1.copy()
xtest_2 = testingMatrix_MindReading2.copy()
xtest_3 = testingMatrix_MindReading3.copy()
ytrain_1 = np.ravel(trainingLabels_MindReading_1.copy())
ytrain_2 = np.ravel(trainingLabels_MindReading_2.copy())
ytrain_3 = np.ravel(trainingLabels_MindReading_3.copy())
xtrain_1 = trainingMatrix_MindReading1.copy()
xtrain_2 = trainingMatrix_MindReading2.copy()
xtrain_3 = trainingMatrix_MindReading3.copy()
yunl_1 = unlabeledLabels_MindReading_1.copy()
yunl_2 = unlabeledLabels_MindReading_2.copy()
yunl_3 = unlabeledLabels_MindReading_3.copy()
xunl_1 = unlabeledMatrix_MindReading1.copy()
xunl_2 = unlabeledMatrix_MindReading2.copy()
xunl_3 = unlabeledMatrix_MindReading3.copy()

#joining the matrix and labels for unlabeled data
xyunl_1 = xunl_1.assign(label=yunl_1)
xyunl_2 = xunl_2.assign(label=yunl_2)
xyunl_3 = xunl_3.assign(label=yunl_3)
##################################################
#Model used for the active learning functions
model = LogisticRegression(solver='liblinear', max_iter=300,random_state=0)

##################################################
#running the Random Active Learning function
MiscTrain,MiscTest,DataSize,iterations = RandomActiveLearning(model,xtrain_1, xtest_1, ytrain_1, ytest_1, xyunl_1)
MiscTrain2,MiscTest2,DataSize2,iterations2 = RandomActiveLearning(model,xtrain_2, xtest_2, ytrain_2, ytest_2, xyunl_2)
MiscTrain3,MiscTest3,DataSize3,iterations3 = RandomActiveLearning(model,xtrain_3, xtest_3, ytrain_3, ytest_3, xyunl_3)
#running the Entropy Active Learning Function
MiscTrainE,MiscTestE,DataSizeE,iterationsE = EntropyActiveLearning(model,xtrain_1, xtest_1, ytrain_1, ytest_1, xyunl_1)
MiscTrain2E,MiscTest2E,DataSize2,iterations2E = EntropyActiveLearning(model,xtrain_2, xtest_2, ytrain_2, ytest_2, xyunl_2)
MiscTrain3E,MiscTest3E,DataSize3E,iterations3E = EntropyActiveLearning(model,xtrain_3, xtest_3, ytrain_3, ytest_3, xyunl_3)
#################################################

##################################################
#saving the data to plot
#MindData Table and data frame - RANDOM ACTIVE LEARNING
table1 = {'Dataset1':MiscTest, 'Dataset2': MiscTest2, 'Dataset3': MiscTest3}
df = pd.DataFrame(table1)
#calculating the average
df['average'] = df.mean(axis=1)

#MindData Table and data frame - ENTROPY ACTIVE LEARNING
table2 = {'Dataset1':MiscTestE, 'Dataset2': MiscTest2E, 'Dataset3': MiscTest3E}
df2 = pd.DataFrame(table2)
#calculating the average
df2['averageENTROPY'] = df2.mean(axis=1)

#Final Plot both active learning algorithm with MindReading data
fig, ax = plt.subplots()  # a figure with one plot
ax.autoscale()
fig.set_dpi(100)
plt.suptitle('Accuracy vs iteration number - MindReading Data',fontsize=12, y=1)
plt.title('Random Learning (blue) and Entropy Learning (orange)', fontsize=10)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(iterations,df['average'])
plt.plot(iterations,df2['averageENTROPY'])
plt.show()
fig.savefig("AccuracyMindReading.jpeg", dpi=300)

##################################################

#MMI dataset
#path to the folder where the data is
pathMMI = "/Users/schama/Desktop/Data/MMI/"
#calling the function to import data
namesMMI = importMatLab(pathMMI)

#saving variables to new names and flattening the labels
ytest_1 = np.ravel(testingLabels_1.copy())
ytest_2 = np.ravel(testingLabels_2.copy())
ytest_3 = np.ravel(testingLabels_3.copy())
xtest_1 = testingMatrix_1.copy()
xtest_2 = testingMatrix_2.copy()
xtest_3 = testingMatrix_3.copy()
ytrain_1 = np.ravel(trainingLabels_1.copy())
ytrain_2 = np.ravel(trainingLabels_2.copy())
ytrain_3 = np.ravel(trainingLabels_3.copy())
xtrain_1 = trainingMatrix_1.copy()
xtrain_2 = trainingMatrix_2.copy()
xtrain_3 = trainingMatrix_3.copy()
yunl_1 = unlabeledLabels_1.copy()
yunl_2 = unlabeledLabels_2.copy()
yunl_3 = unlabeledLabels_3.copy()
xunl_1 = unlabeledMatrix_1.copy()
xunl_2 = unlabeledMatrix_2.copy()
xunl_3 = unlabeledMatrix_3.copy()
#joining the matrix and labels for unlabeled data
xyunl_1 = xunl_1.assign(label=yunl_1)
xyunl_2 = xunl_2.assign(label=yunl_2)
xyunl_3 = xunl_3.assign(label=yunl_3)
##################################################

##################################################

#random active learning
MiscTrain4,MiscTest4,DataSize4,iterations4 = RandomActiveLearning(model,xtrain_1, xtest_1, ytrain_1, ytest_1, xyunl_1)
MiscTrain5,MiscTest5,DataSize5,iterations5 = RandomActiveLearning(model,xtrain_2, xtest_2, ytrain_2, ytest_2, xyunl_2)
MiscTrain6,MiscTest6,DataSize6,iterations6 = RandomActiveLearning(model,xtrain_3, xtest_3, ytrain_3, ytest_3, xyunl_3)

#entropy active learning
MiscTrain4E,MiscTest4E,DataSize4E,iterations4E = EntropyActiveLearning(model,xtrain_1, xtest_1, ytrain_1, ytest_1, xyunl_1)
MiscTrain5E,MiscTest5E,DataSize5E,iterations5E = EntropyActiveLearning(model,xtrain_2, xtest_2, ytrain_2, ytest_2, xyunl_2)
MiscTrain6E,MiscTest6E,DataSize6E,iterations6E = EntropyActiveLearning(model,xtrain_3, xtest_3, ytrain_3, ytest_3, xyunl_3)
##################################################


##################################################
#Saving the data to plot
#MMI Table and data frame - Random Learning
table1 = {'MMI1':MiscTest4, 'MMI2': MiscTest5, 'MMI3': MiscTest6}
df = pd.DataFrame(table1)
#Calculating average
df['average'] = df.mean(axis=1)

#MMI Table and data frame - Entroby-based Learning
table2 = {'MMI1':MiscTest4E, 'MMI2': MiscTest5E, 'MMI3': MiscTest6E}
df2 = pd.DataFrame(table2)
#Calculating average
df2['averageENTROPY'] = df2.mean(axis=1)

#saving the results to a file
fig, ax = plt.subplots()  # a figure with one plot
ax.autoscale()
fig.set_dpi(100)
plt.suptitle('Accuracy vs iteration number - MMI Data',fontsize=12, y=1)
plt.title('Random Learning (blue) and Entropy Learning (orange)', fontsize=10)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(iterations,df['average'])
plt.plot(iterations,df2['averageENTROPY'])
plt.show()
fig.savefig("AccuracyMMI.jpeg", dpi=300)
##################################################


