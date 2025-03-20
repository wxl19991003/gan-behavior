import pandas as pd
import copy
b2data=pd.read_csv("log/r_bpi2012.csv",low_memory=False,index_col=0)
b3data=pd.read_csv("log/rawdata2013.csv",low_memory=False,index_col=0)
ldata=pd.read_csv("log/large_log.csv", low_memory=False, index_col=0)
sdata=pd.read_csv("log/short_log.csv", low_memory=False, index_col=0)

b2data=b2data.values
b3data=b3data.values
ldata=ldata.values
sdata=sdata.values

def nptolist(npdata):
    l1 = []
    n = 0
    for i in npdata:
        l2 = []
        for j in i:
            if isinstance(j, str):
                l2.append(j)
        l1.append(l2)
        n += 1
    print("***********")
    return l1

b2data=nptolist(b2data)
b3data=nptolist(b3data)
ldata=nptolist(ldata)
sdata=nptolist(sdata)

def geteventlist(data):
    l=[]
    tag=0
    for i in data:
        if len(i)>tag:
            tag=len(i)
        for j in i:
            if j not in l:
                l.append(j)
    return l,len(l),tag

print(geteventlist(b2data))
print(geteventlist(b3data))

print(geteventlist(ldata))
print(geteventlist(sdata))