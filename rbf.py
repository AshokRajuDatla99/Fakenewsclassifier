import numpy as np
import math

def rbf1(x,y,z):
    f1 = math.pow(((x - 3.75)/0.5),2)
    f2 = math.pow(((y - 95) / 10), 2)
    f3 = math.pow(((z - 14.5)/3),2)

    sum = f1 + f2 + f3
    sqrt = math.sqrt(sum)
    return math.exp(-(sqrt))

def rbf2(x,y,z):
    f1 = math.pow(((x - 3.25)/0.5),2)
    f2 = math.pow(((y - 82.5) / 15), 2)
    f3 = math.pow(((z - 12)/2),2)

    sum = f1 + f2 + f3
    sqrt = math.sqrt(sum)
    return math.exp(-(sqrt))

def rbf3(x,y,z):
    f1 = math.pow(((x - 2.9)/0.2),2)
    f2 = math.pow(((y - 65) / 15), 2)
    f3 = math.pow(((z - 10)/2),2)

    sum = f1 + f2 + f3
    sqrt = math.sqrt(sum)
    return math.exp(-(sqrt))

def rbf4(x,y,z):
    f1 = math.pow(((x - 1.4)/2.8),2)
    f2 = math.pow(((y - 27.5) / 55), 2)
    f3 = math.pow(((z - 6.5)/5),2)

    sum = f1 + f2 + f3
    sqrt = math.sqrt(sum)
    return math.exp(-(sqrt))

def rbf(x,y,z):
    l = []
    l.append(rbf1(x,y,z))
    l.append(rbf2(x, y, z))
    l.append(rbf3(x, y, z))
    l.append(rbf4(x, y, z))

    return l

# the data mentioned
student_data = [[3.2,67,1220],
                [3.7,88,1360],
                [2.9,76,980],
                [3.3,45,880],
                [4.0,99,1520],
                [3.6,91,1410],
                [2.6,82,1020],
                [2.5,54,640],
                [2.8,73,1140],
                [3.1,77,1160]]

student_data = np.array(student_data)

last_col = np.divide(student_data[:,2],100)
student_data[:,2] = last_col

output_rbf = []
for i in range(len(student_data)):
    x = student_data[i][0]
    y = student_data[i][1]
    z = student_data[i][2]
    output_rbf.append(rbf(x,y,z))

output_rbf = np.array(output_rbf)
print(output_rbf)

def wta(output_rbf):

    for row in output_rbf:
        while not all(x == 1 or x == 0 for x in row):
            for i in range(len(row)):
                row[i] = row[i]*1.2 - 0.2*(np.sum(row)-row[i])
                if row[i]<0:
                    row[i] = 0
                elif row[i]>1:
                    row[i] = 1
    print(output_rbf)

wta(output_rbf)

category = []
for i in output_rbf:
    result = np.where(i == 1)
    category.append(result[0][0] + 1)
print(category)

import matplotlib.pyplot as plt
x=[]
for i in range(1,11):
    x.append(i)
plt.scatter(x, category)

plt.xlabel('student number')
plt.ylabel('category')

plt.title('STUDENT CLASSIFIER')
plt.show()