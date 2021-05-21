import numpy as np
import random
import math
import matplotlib.pyplot as plt

temp = 10000
cooling_rate = 0.005
error = []
iterations = []

# define the distance matrix
#  campuses selected : abington, altoona, berks
# erie,great valley, hazleton, scranton
# shenango,university_park, York

distance_matrix = [[0,234,62,414,26,95,119,363,190,100],
                   [234,0,184,204,216,155,192,150,43,159],
                   [62,184,0,353,42,54,102,301,141,55],
                   [414,204,353,0,385,315,316,94,213,350],
                   [26,216,42,385,0,101,126,346,168,78],
                   [95,155,54,315,101,0,44,262,113,110],
                   [119,192,102,316,126,44,0,299,150,153],
                   [363,150,301,94,346,262,299,0,161,288],
                   [190,43,141,213,168,113,150,161,0,113],
                   [100,159,55,350,78,110,153,288,113,0]]
distance_matrix = np.array(distance_matrix)
distance_matrix = np.divide(distance_matrix,500)
print(np.mean(distance_matrix))
print(distance_matrix)

# generate random tour matrix
# adding the 11th row to make sure start point and end point is same
tour_matrix = np.random.randint(0,2,size=(10,10))
tour_matrix = np.column_stack((tour_matrix,tour_matrix[:,0]))


def store_best_tour(arr):
    p = arr[:,0] + arr[:,10]
    tup = np.where(p==2)
    if 1 in p or len(tup[0]>1):
        return
    else:
        best_tour = arr

def cost_function(arr):
    #skip penalty as penalty1
    penalty1 = 0
    rows,cols = arr.shape
    count = np.count_nonzero(arr==1)
    # print(count)
    if count-rows < 0:
        penalty1 = abs(count - rows)* 10
    # print(penalty1)

    #duplicate penalty as penalty2
    sum_cols = arr.sum(axis=0)
    sum_rows = arr.sum(axis=1)

    penalty2 =0
    for i in sum_rows:
        penalty2 += (i-1)*(i-2)* 10 #modified formula
    for i in sum_cols:
        penalty2 += (i-1)*(i-2)* 10 #modified formula
    # print(penalty2)

    #distance penalty as penalty3
    penalty3 = 0
    for i in range(cols-1):
        for j in range(rows):
            if arr[j,i]!=0:
                for k in range(rows):
                    if arr[k,i+1]!=0 and j!=k:
                        penalty3 = penalty3 + distance_matrix[j,k]
    # print(penalty3)
    total_cost = penalty1 + penalty2 + penalty3
    return total_cost

cost = cost_function(tour_matrix)
# runs for 1838 iterations
count = 0
while temp>1:
    # print(cost)
    x = random.randint(0,9)
    y = random.randint(0,9)
    flag = 0
    if tour_matrix[x][y]:
        tour_matrix[x][y] = 0
        if (y == 0):
            tour_matrix[x][10] = 0
    else:
        tour_matrix[x][y] = 1
        if (y == 0):
            tour_matrix[x][10] = 1
        flag=1

    new_cost = cost_function(tour_matrix)
    change = cost-new_cost
    error.append(change)
    count +=1
    iterations.append(count)

    prob = 1/(1+math.exp(-change/temp))
    # print(prob)
    # print()

    if(prob > random.random()):
        cost = new_cost
        store_best_tour(tour_matrix)
    elif(flag==0):
        tour_matrix[x][y]=1
        if(y==0):
            tour_matrix[x][10]=1
    elif(flag==1):
        tour_matrix[x][y]=0
        if (y == 0):
            tour_matrix[x][10] = 0

    temp *= (1-cooling_rate)

print(tour_matrix)


plt.plot(iterations, error)

plt.xlabel('Temperature')
plt.ylabel('Error')

plt.title('TSP')
plt.show()
