import numpy as np
import math
import collections

# dataset
C = [[0,0,0,0,0,0,0,0],
     [0,0,0,1,1,0,0,0],
     [0,0,1,0,0,1,0,0],
     [0,1,0,0,0,0,1,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,1,0],
     [0,0,1,0,0,1,0,0],
     [0,0,0,1,1,0,0,0],
     [0,0,0,0,0,0,0,0]]

D = [[0,0,0,0,0,0,0,0],
     [0,1,1,1,1,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,1,0,0,0,0,1,0],
     [0,1,0,0,0,0,1,0],
     [0,1,0,0,0,0,1,0],
     [0,1,0,0,0,0,1,0],
     [0,1,0,0,0,1,0,0],
     [0,1,1,1,1,0,0,0],
     [0,0,0,0,0,0,0,0]]

F = [[0,0,0,0,0,0,0,0],
     [0,1,1,1,1,1,1,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,1,1,1,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

I = [[0,0,0,0,0,0,0,0],
     [0,1,1,1,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,1,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

X = [[0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,1,0,1,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,1,0,1,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0]]

Y = [[0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,1,0,1,0,0,0],
     [0,0,1,0,1,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

c = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,1,0,0,1,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,1,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

d = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,1,0,0,0],
     [0,0,0,0,1,0,0,0],
     [0,0,0,0,1,0,0,0],
     [0,0,1,1,1,0,0,0],
     [0,1,0,0,1,0,0,0],
     [0,1,0,0,1,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

f = [[0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [1,1,1,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

i = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]

x = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,1,0,1,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,1,0,1,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0]]

y = [[0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,1,0,0,0,1,0,0],
     [0,0,1,0,1,0,0,0],
     [0,0,0,1,0,0,0,0],
     [0,0,1,0,0,0,0,0],
     [0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]


def neighbours(char, i, j):
    sum = 0
    for k in range(i - 1, i + 2):
        for l in range(j - 1, j + 2):
            sum += char[k][l]

    return sum


def tp_joint(char, i, j):
    sum = char[i - 1][j] + char[i + 1][j] + char[i][j - 1] + char[i][j + 1]
    if sum >= 3:
        return 1
    return 0

def aspect_ratio(char):

    max_length = 0
    for i in range(1, len(char)-1):
        length = 0
        start = 0
        end = 0
        for j in range(1,len(char[0])-1):
            if char[i][j] == 1 and start == 0:
                start = j

            elif char[i][j]==1 and start!=0:
                end = j
        if end > start:
            length = end - start  + 1
        if (length > max_length):
            max_length = length

    return max_length

def col_pixel_density(char):

    total_pixels = 0
    mid_pixels = 0
    for i in range(1,len(char[0])-1):
        counter = collections.Counter(char[:,i])
        if 1 in counter.keys():
            total_pixels +=counter[1]

        if i in (1,2,3):
            mid_pixels +=counter[1]


    return mid_pixels/total_pixels


def loop(char):
    # 8 cols and 10 rows
    two_nei = 0
    tj = 0
    for i in range(1, len(char) - 1):
        for j in range(1, len(char[i]) - 1):
            if char[i][j] != 0:
                nei = neighbours(char, i, j)
                tj += tp_joint(char, i, j)

                if (nei == 2):
                    two_nei += 1

    return two_nei,tj

# initialising charactersets
capital_letters = {'C':np.array(C),'D':np.array(D),'F':np.array(F),'I':np.array(I),'X':np.array(X),'Y':np.array(Y)}
small_letters = {'c':np.array(c),'d':np.array(d),'f':np.array(f),'i':np.array(i),'x':np.array(x),'y':np.array(y)}

kernel_tn = []
kernel_tj = []
kernel_ar = []
kernel_pd = []
kernel_data = []
test_tn = []
test_tj = []
test_ar = []
test_pd = []
test_data=[]

for arr in capital_letters.values():
    two_nei,tj = loop(arr)
    aratio = aspect_ratio(arr)
    pix_conc = col_pixel_density(arr)
    kernel_data.append((two_nei, tj, aratio, pix_conc))
    kernel_pd.append(pix_conc)
    kernel_tn.append(two_nei)
    kernel_tj.append(tj)
    kernel_ar.append(aratio)

# print(kernel_pd)

for arr in small_letters.values():
    two_nei,tj = loop(arr)
    aratio = aspect_ratio(arr)
    pix_conc = col_pixel_density(arr)

    test_data.append((two_nei,tj,aratio,pix_conc))

    test_pd.append(pix_conc)
    test_tn.append(two_nei)
    test_tj.append(tj)
    test_ar.append(aratio)

# print(test_pd)

def normalize(w,x,y,z):

    max_w = max(w)
    max_x = max(x)
    max_y = 8
    list = []
    for i in range(len(x)):
        list.append((w[i]/max_w,x[i]/max_x,y[i]/max_y,z[i]))

    return list

test_data = normalize(test_tn,test_tj,test_ar,test_pd)
kernel_data = normalize(kernel_tn,kernel_tj,kernel_ar,kernel_pd)

print("Kernel-Data or Capital letters data")
print("Char C: ",kernel_data[0])
print("Char D: ",kernel_data[1])
print("Char F: ",kernel_data[2])
print("Char I: ",kernel_data[3])
print("Char X: ",kernel_data[4])
print("Char Y: ",kernel_data[5])

print("")
print("Test-Data or Small letters data")
print("Char c: ",test_data[0])
print("Char d: ",test_data[1])
print("Char f: ",test_data[2])
print("Char i: ",test_data[3])
print("Char x: ",test_data[4])
print("Char y: ",test_data[5])



def kernel_function(test_tuple,kernel_tuple):

    prob1 = 0
    prob2 = 0
    prob3 = 0
    prob4 = 0
    for i in range(len(test_tuple)):
        if i==0:
            prob1 += 3* math.exp(-(math.pow(test_tuple[i]-kernel_tuple[i],2)/2))

        if i==1:
            prob2 += 1 * math.exp(-(math.pow(test_tuple[i] - kernel_tuple[i], 2) / 2))

        if i==2:
            prob3 += 2 * math.exp((math.pow(test_tuple[i] - kernel_tuple[i], 2) / 2))

        if i==3:
            prob4 += 4 * math.exp(-(math.pow(test_tuple[i] - kernel_tuple[i], 2) / 2))


    print("probability: %s,%s,%s,%s"%(prob1,prob2,prob3,prob4))
    prob = prob1 + prob2 + prob3 + prob4
    return prob/10


def pnn(kernel_data,test_data):

    probabilities = []
    for test_tuple in test_data:
        temp = []
        print(" \nFor each tuple: ",test_tuple)
        for kernel_tuple in kernel_data:
            # print(kernel_function(test_tuple,kernel_tuple))
            temp.append(kernel_function(test_tuple,kernel_tuple))
        probabilities.append(temp)

    # print(probabilities)
    return probabilities
prob_list = pnn(kernel_data,test_data)


def categorize(l):
    for i in l:
        print("category: %s" %i.index(max(i)))
categorize(prob_list)

