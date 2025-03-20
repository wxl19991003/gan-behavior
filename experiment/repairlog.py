import pandas as pd
import copy

def repair(eventlist,label,testdata):
    l0=[]
    for i in range(len(label)):
        l1 = []
        if label[i]==0:
            l1.append(eventlist[label[i]])
            for j in testdata[i]:
                l1.append(j)
        else:
            tag=label[i]
            for j in range(tag):
                l1.append(testdata[i][j])
            l1.append(eventlist[tag])
            for j in range(len(label)-tag):
                l1.append(testdata[i][j+tag])
        l0.append(copy.deepcopy(l1))

'''
test
'''

l=[0,1,2,3,4,6,7,8,9]
tag=5
lx=[]
for j in range(tag):
    lx.append(l[j])
lx.append(tag)
for j in range(len(l) - tag):
    lx.append(l[j + tag])
print(lx)