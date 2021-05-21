import pandas as pd
from math import sqrt
from math import pi
from math import exp
import numpy as np

def check_categorical_data(df):  #used to define categorical and numeircal columns
    cd_col = {} #dictionary of categorical columns and unique values
    n_col = [] #list of numerical columns in the dataframe
    for i in col[:-1]:
        if len(pd.unique(df[i])) < 4:    #hardcoded instance
            cd_col[i]=list(pd.unique(df[i]))
        else:
            n_col.append(i)
    return cd_col,n_col

def create_prob_dict(c):
    dict={}
    for i in c:
        dict[i]=[]
    return dict

def modify_probs(prob_d):
    modified_prob = create_prob_dict(categorical_columns)
    for keys,values in categorical_columns.items():
        d = {}
        for i in range(len(values)):
            d[categorical_columns[keys][i]] = prob_d[keys][i]
        modified_prob[keys] = d
    return modified_prob

def laplacian_correction(df): #laplacian smoothing for categorical data
    prob_dict = create_prob_dict(categorical_columns)
    for i,value in categorical_columns.items():
        for j in value:
            prob = (len(df.loc[df[i]==j]) + 1) / (len(df) + len(categorical_columns)*2)
            prob_dict[i].append(prob)
    return prob_dict

def categorical_prob(uni,key,nested_dict): # a nested dictionary of class 1 and class -1: with separate dictionaries consisting unique categories as keys and probabilities as values
    return nested_dict[key][uni]

def compute_cat_class(df):
    predicted_class = []
    for index, row in df.iterrows():
        cprob_yes = float(len(class1_train_data) / len(train_data))
        cprob_no = float(len(class2_train_data) / len(train_data))
        for i in col[:-1]:
            cprob_yes *= categorical_prob(row[i],i,modified_probs_class1)
            cprob_no *= categorical_prob(row[i], i, modified_probs_class2)

        if (cprob_yes > cprob_no):
            predicted_class.append(1)
        else:
            predicted_class.append(-1)

    return predicted_class

def accuracy_metrics(predicted_class,original_class): #return's accuracy and confusion matrix
    count = 0
    confusion_matrix = [0,0,0,0]
    for i in range(len(original_class)):
        if(original_class[i]==predicted_class[i] and original_class[i]==1):
            confusion_matrix[0] = confusion_matrix[0] + 1
        if (original_class[i] == predicted_class[i] and original_class[i] == -1):
            confusion_matrix[1] = confusion_matrix[1] + 1
        if (original_class[i] != predicted_class[i] and original_class[i] == 1):
            confusion_matrix[2] = confusion_matrix[2] + 1
        if (original_class[i] != predicted_class[i] and original_class[i] == -1):
            confusion_matrix[3] = confusion_matrix[3] + 1

        if(predicted_class[i]==original_class[i]):
            count = count+1
    accuracy = count/len(original_class)
    
    return accuracy,confusion_matrix

def guassian_prob(x,mean,stdev): #guassian probability for numerical data
    const1 = float(1/sqrt(2*pi))
    const2 = float(1/stdev)
    const = const1 * const2
    n = - ((x - mean)**2)
    d = 2*(pow(stdev,2))
    ep = exp(n/d)
    return const*ep

def compute_class(df): #finding the guassian probability of test data using the earler computed summaries(mean,stdev)
    predicted_class = []
    for index, row in df.iterrows():
        cprob_yes = float(len(class1_train_data) / len(train_data))
        cprob_no = float(len(class2_train_data) / len(train_data))
        for i in col[:-1]:
            cprob_yes *= guassian_prob(row[i],summaries_class1[i][0],summaries_class1[i][1])
            cprob_no *= guassian_prob(row[i],summaries_class2[i][0],summaries_class2[i][1])

        if(cprob_yes>cprob_no):
            predicted_class.append(1)
        else:
            predicted_class.append(-1)

    return predicted_class

def stats(df):
    dict={}
    for i in col[:-1]:
        if i not in dict:
            dict[i] = (df[i].mean(),df[i].std())
    return dict

def create_dict(train_data):
    dict = {1:[],-1:[]}
    for i in range(len(train_data)):
        if train_data.iloc[i,-1]==1:
            dict[1].append(list(train_data.iloc[i]))
        else:
            dict[-1].append(list(train_data.iloc[i]))
    return dict

#loading test and train data
train_data = pd.read_csv("buyTraining.txt",delimiter=" ",header=None)
col = np.arange(len(train_data.columns))
train_data.columns = col
test_data = pd.read_csv("buyTesting.txt",delimiter=" ",header=None)
test_data.columns = col

# list of categorical columns and numerical columns
categorical_columns,numerical_columns = check_categorical_data(train_data)  #it will return a dict of categorical columns and list of numerical columns

# separating the class data into two separate dataframes
class_separated_dict = create_dict(train_data)
class1_train_data = pd.DataFrame(class_separated_dict[1],columns=col)    #pandas dataframe for class 1
class2_train_data = pd.DataFrame(class_separated_dict[-1],columns=col)   #pandas dataframe for class 2
original_class = test_data.iloc[:, -1].to_list()

# pass only numerical cols dataframe
if(len(numerical_columns)!=0):
    # computing dictionary with feature:(mean,stdev) key-value pairs.
    overall_summaries = stats(train_data)
    summaries_class1 = stats(class1_train_data)
    summaries_class2 = stats(class2_train_data)
    predicted_class = compute_class(test_data.iloc[:, :-1])

#pass only categorical cols dataframe
if(len(categorical_columns)!=0):
    # function to do the laplacian correction and return the probabilities dict with column name and unique categories probability
    probs_class1 = laplacian_correction(class1_train_data)
    probs_class2 = laplacian_correction(class2_train_data)

    print(probs_class1)

    # function to create a nested dictionary with column name as key and values is a dict of unique value and prob pairs
    modified_probs_class1 = modify_probs(probs_class1)
    modified_probs_class2 = modify_probs(probs_class2)
    predicted_class = compute_cat_class(test_data.iloc[:,:-1])

accuracy,confusion_matrix = accuracy_metrics(predicted_class,original_class)

print(accuracy)
print("The true  positives(tp):" + str(confusion_matrix[0]))
print("The true  negatives(tp):" + str(confusion_matrix[1]))
print("The false positives(tp):" + str(confusion_matrix[2]))
print("The false negatives(tp):" + str(confusion_matrix[3]))

