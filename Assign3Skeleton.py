
#A skeleton for implementing Naive Bayes Classifier in Python.
## Author: Md Faisal Kabir

import pandas as pd
import random
import time

# trainingFile = "irisTraining.txt"
# testingFile = "irisTesting.txt"

train_data = pd.read_csv("irisTraining.txt")
# test_data = pd.read_csv(testingFile)

print(train_data)

#Training... Collect mean and standard deviation for each dimension for each class..
#Also, calculate P(C+) and P(C-)

#Testing .....
# Xtest = numpy.loadtxt(testingFile)
# nn = Xtest.shape[0] # Number of points in the testing data.
#
# tp = 0 #True Positive
# fp = 0 #False Positive
# tn = 0 #True Negative
# fn = 0 #False Negative

#Iterate over all points in testing data
  #For each point find the P(C+|Xi) and P(C-|Xi) and decide if the point belongs to C+ or C-..
  #Recall we need to calculate P(Xi|C+)*P(C+) ..
  #P(Xi|C+) = P(Xi1|C+) * P(Xi2|C+)....P(Xid|C+)....Do the same for P(Xi|C-)
  #Now that you've calculate P(Xi|C+) and P(Xi|C-), we can decide which is higher 
  #P(Xi|C-)*P(C-) or P(Xi|C-)*P(C-) ..
  #increment TP,FP,FN,TN accordingly, remember the true lable for the ith point is in Xtest[i,d]

#}

#Calculate all the measures required..
 
