import numpy as np
import math

v = np.random.random(size=(2,2)) #initial weigths
w = np.random.random(size=(1,3)) #hidden weigths
print("old")
print('matrix V:')
print(v)
print()
print('matrix W:')
print(w)
l = 0.4 #learning rate
train_data = []
output_data = []
z = []
for i in range(0,181,2):
    rad = math.pi * (i/180)
    train_data.append(rad)
    output_data.append(round(math.cos(rad),5))

# data normalisation and modification
norm = max(train_data)
for i in range(len(train_data)):
    train_data[i] = train_data[i]/norm
    output_data[i] = (output_data[i]+1)/2

def energy_func(t,y4):
    return 0.5 * math.pow(t-y4,2)

def act(val):
    return 1/(1+math.exp(-val))

flag = True
while(flag):
    count = 0
    error = 0
    w02, w03, w04, w12, w13, w24, w34 = [0.0 for i in range(7)]

    for j in range(len(train_data)):
        t = output_data[j]
        # feed forward
        x1 = train_data[j]
        y2 = act(v[1][0]*x1 + v[0][0])
        y3 = act(v[1][1]*x1 + v[0][1])
        y4 = act(w[0][0]*y2 + w[0][1]*y3 + w[0][2])
        error += energy_func(t,y4)

        #backward
        d4 = (t-y4)*y4*(1-y4)
        d2 = y2*(1-y2)*d4*w[0][0]
        d3 = y3 * (1 - y3) * d4 * w[0][1]

        #updating weigths
        w02 += l*d2
        w03 += l*d3
        w04 += l*d4
        w12 += l*d2*x1
        w13 += l*d3*x1
        w24 += l*d4*y2
        w34 += l*d4*y3

    v[0][0] += w02
    v[0][1] += w03
    v[1][0] += w12
    v[1][1] += w13
    w[0][0] += w24
    w[0][1] += w34
    w[0][1] += w04

    if error < 0.02:
        flag = False

print()
print()
print("new")
print('matrix V:')
print(v)
print()
print('matrix W:')
print(w)

new_train = []
out = []
for i in range(182,201,2):
    rad = math.pi * (i / 180)
    new_train.append(rad)
    out.append(round(math.cos(rad), 5))

norm = max(train_data)
for i in range(len(new_train)):
    new_train[i] = new_train[i]/norm
    out[i] = (out[i]+1)/2

new_output = []

for j in range(len(new_train)):
    # feed forward
    x1 = new_train[j]
    y2 = act(v[1][0] * x1 + v[0][0])
    y3 = act(v[1][1] * x1 + v[0][1])
    y4 = act(w[0][0] * y2 + w[0][1] * y3 + w[0][2])
    new_output.append(y4)

# print(train_data)
# print(output_data)
# print(new_output)

import matplotlib.pyplot as plt
figure, axis = plt.subplots(2)
figure.suptitle('BACKPROPAGATION FOR FUNCTION APPROXIMATION')
axis[0].plot(new_train, out)

# axis[0].set_title("Original Cos(x)")
axis[1].plot(new_train,new_output)
# axis[1].set_title("Approximated Cos(x)")
plt.xlabel("Radians")
plt.ylabel("Values")
plt.show()