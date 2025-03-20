from random import randint
import pandas as pd
import numpy as np
import pprint
import copy
from sklearn import model_selection
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import dataloader, dataset, TensorDataset
import torch
from time import *
pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("log/helpdesk_pre.csv", low_memory=False, index_col=0)
npdata = dataframe.values

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


def geteventlist(l):
    eventlist = []
    for i in l:
        for j in i:
            if j not in eventlist:
                eventlist.append(j)
    return eventlist, len(eventlist)


def gettraceevent(trace):
    l = []
    for i in trace:
        if i not in l:
            l.append(i)
    return l


def initbkanktrace(eventistlen):
    l = []
    for i in range(eventistlen):
        l.append(0)
    return l


def inittaglist(eventistlen):  # 初始化tag
    l = []
    for i in range(eventistlen):
        j = []
        for i in range(eventistlen):
            j.append(0)
        l.append(j)
    return l


def delevent(data):  # 删除活动并返回被删除的活动以及它直接前继与后继
    l1 = data.copy()
    l2 = []
    delindexlist=[]
    for i in l1:
        length = len(i)
        r = randint(0, length - 1)
        l2.append(i[r])
        delindexlist.append(r)
        del i[r]
    return l1, l2,delindexlist


def getallindex(l1, m):  # 获得列表中所有元素的下标,l1是列表，m是要查的元素
    length = len(l1)
    n = l1.count(m)
    indexn = []
    index = 0
    for i in list(range(n)):
        index = l1.index(m, index, length)
        indexn.append(index)
        index += 1
    return indexn


def getorder(log, eventlist, eventlistlen):
    l = eventlist
    ll = inittaglist(eventlistlen)
    for i in range(eventlistlen):
        t1 = []
        for trace in log:
            if l[i] in trace:
                t2 = []
                t1 = getallindex(trace, l[i])
                l1 = gettraceevent(trace)  # 临时列表，储存特定迹的事件
                for j in l1:
                    if l[i] != j:
                        t2 = getallindex(trace, j)
                        index = l.index(j)
                        if max(t1) < min(t2):
                            if ll[i][index] != 2 and ll[index][i] != 2:
                                ll[i][index] = 1
                                ll[index][i] = 1
                        elif max(t1) > min(t2) and min(t1) < max(t2):
                            ll[i][index] = 2
                            ll[index][i] = 2
                        elif max(t1) > max(t2) and ll[i][index] == 1:  # 前面俩个是迹内判断，这个是迹间判断，
                            # 虽然放在同一级循环内，但是逻辑是俩层判断才形成的迹间判断
                            ll[i][index] = 2
                            ll[index][i] = 2
                    else:
                        number = trace.count(j)
                        if number == 1 and ll[i][i] != 2:
                            ll[i][i] = 3
                        else:
                            ll[i][i] = 2
        t1.clear()
    for i in range(len(ll)):
        for j in range(len(ll)):
            if ll[i][j] == 0:
                ll[i][j] = 3
                print("改了")
    return ll


def getconcurrencegraph(ll):
    lc = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 2:
                lc[i][j] = 0
            else:
                lc[i][j] = 1
    return lc


def getqequencegraph(ll):
    ls = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 1:
                ls[i][j] = 0
    return ls


def getexclusivegraph(ll):
    le = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 3:
                le[i][j] = 0
            else:
                le[i][j] = 1
    return le


def expandlist(l, eventlistl, blanklist):
    tblanklist = blanklist.copy()
    lenth = len(eventlistl)
    for i in range(14-lenth):
        l.append(tblanklist)
    return l


def getbehaviorgraph(li, l11, lc, le, ls, blanklist):
    ms = []
    mc = []
    me = []
    molc = lc.copy()
    mole = le.copy()
    mols = ls.copy()
    tblanklist = blanklist.copy()
    for i in l11:
        eventl = gettraceevent(i)
        tracelc = molc.copy()
        tracele = mole.copy()
        tracels = mols.copy()
        for j in li:
            if j not in eventl:
                n = li.index(j)
                tracelc[n] = tblanklist
                tracele[n] = tblanklist
                tracels[n] = tblanklist
                # for j in range(eventlistlen):
                #     tracelc[n][j]=0
                #     tracele[n][j]=0
                #     tracels[n][j]=0
        # tracels=expandlist(tracels,li,blanklist)
        # tracele=expandlist(tracele,li,blanklist)
        # tracelc=expandlist(tracelc,li,blanklist)
        ms.append(tracels)
        me.append(tracele)
        mc.append(tracelc)
    return ms, me, mc


def getsln1(trace, eventlist,delindex):
    sln = []
    length = len(trace)
    for i in range(length):
        line = []
        for j in range(len(eventlist)):
            line.append(0)
        l = trace[:i + 1]
        for e in eventlist:
            line[eventlist.index(e)] = l.count(e)
        sln.append(line)
    l1 = []
    for j in range(len(eventlist)):
        l1.append(0)
    for i in range(14-len(trace) ):
        sln.append(l1)
    return sln


def getsln(trace, eventlist,datatraceindex):
    sln = []
    length = len(trace)
    tag=0
    l1 = []
    for j in range(len(eventlist)):
        l1.append(0)
    for i in range(length):
        if i != datatraceindex:
            line = []
            for j in range(len(eventlist)):
                line.append(0)
            l = trace[:i + 1]
            for e in eventlist:
                line[eventlist.index(e)] = l.count(e)
            sln.append(line)
        else:
            if datatraceindex !=0:
                h=copy.deepcopy(l1)
                sln.append(h)
            else:
                sln.append(l1)
            tag=1
    l = []
    lengthx=len(sln)
    for j in range(len(eventlist)):
        l.append(0)
    for i in range(14-lengthx):
        sln.append(l)
    return sln


def getslsc(trace, eventlist, blanklist):
    length = len(eventlist)  # 这是获取关系，所以不关心迹的长度
    sls = inittaglist(length)
    slc = inittaglist(length)
    el = gettraceevent(trace)
    for e in el:
        l1 = getallindex(trace, e)
        for j in el:
            if j != e:
                l2 = getallindex(trace, j)
                if max(l1) < min(l2):
                    sls[eventlist.index(e)][eventlist.index(j)] = 1
                    sls[eventlist.index(j)][eventlist.index(e)] = 1
                else:
                    if sls[eventlist.index(e)][eventlist.index(j)] != 1:
                        slc[eventlist.index(e)][eventlist.index(j)] = 1
                        slc[eventlist.index(j)][eventlist.index(e)] = 1
            else:
                if trace.count(e) != 1:
                    slc[eventlist.index(e)][eventlist.index(j)] = 1
                    slc[eventlist.index(j)][eventlist.index(e)] = 1
                # else:
                #     sls[eventlist.index(e)][eventlist.index(j)]=1
                #     sls[eventlist.index(j)][eventlist.index(e)]=1
    # sls=expandlist(sls,eventlist,blanklist)
    # slc=expandlist(slc,eventlist,blanklist)
    return sls, slc


def getmultigraph(das, eventlist, lc, le, ls, blanklist,delindex):
    actgraph = []
    lln = []
    lls = []
    llc = []
    multifeaturelist1 = []
    multifeaturelist2 = []
    multifeaturelist3 = []
    for i in range(len(das)):
        sls, slc = getslsc(das[i], eventlist, blanklist)
        lls.append(sls)
        llc.append(slc)
        sln = getsln(das[i], eventlist,delindex)
        lln.append(sln)
    ms, me, mc = getbehaviorgraph(eventlist, das, lc, le, ls, blanklist)
    print(len(lln[0]), len(lls[0]), len(llc[0]), len(ms[0]), len(me[0]), len(mc[0]))
    print(len(lln[0][0]), len(lls[0][0]), len(llc[0][0]), len(ms[0][0]), len(me[0][0]), len(mc[0][0]))
    print("*************************************************************************")
    for i in range(len(das)):
        multifeature1 = np.array([lln[i], lls[i], llc[i], ms[i], me[i], mc[i]])
        multifeature2 = np.array([lln[i]])
        multifeature3 = np.array([lln[i], lls[i], llc[i]])
        multifeaturelist1.append(multifeature1)
        multifeaturelist2.append(multifeature2)
        multifeaturelist3.append(multifeature3)
    multifeaturelist1 = np.array(multifeaturelist1)
    multifeaturelist2 = np.array(multifeaturelist2)
    multifeaturelist3 = np.array(multifeaturelist3)
    print("multifeature_length:", len(multifeaturelist1))
    return multifeaturelist1, multifeaturelist2, multifeaturelist3


class CNN_Net1(nn.Module):
    def __init__(self):
        super(CNN_Net1, self).__init__()
        self.conv1 = nn.Conv2d(6, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5408, 100)
        self.fc2 = nn.Linear(100, 14)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
        x = x.view(-1, 5408)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def getstr(self):
        return 'CNN_Net1'


class CNN_Net2(nn.Module):
    def __init__(self):
        super(CNN_Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5408, 100)
        self.fc2 = nn.Linear(100, 14)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
        x = x.view(-1,5408)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def getstr(self):
        return 'CNN_Net2'


class CNN_Net3(nn.Module):
    def __init__(self):
        super(CNN_Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5408, 100)
        self.fc2 = nn.Linear(100, 14)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
        x = x.view(-1,5408)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def getstr(self):
        return 'CNN_Net3'


######################  17

def labeltotorch_tensor(l, eventlist):
    labellist = []
    for i in l:
        labellist.append(eventlist.index(i))
    return labellist


def trainandtest(train_loader,test_loader,cnn,loss_func,lrc,train_dataset,epochs,device,ix):
    model=cnn().to(device)
    opt= torch.optim.SGD(model.parameters(), lr=lrc, momentum=0.9)
    trainloss_count = []
    for epoch in range(epochs):
        running_loss = 0
        running_acc = 0
        for i, (x, y) in enumerate(train_loader):
            # batch_x = Variable(x)
            # batch_y = Variable(y)
            batch_x = x.to(device)
            batch_y = y.to(device)
            # 获取最后输出
            out = model(batch_x)  # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out, batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上
            running_loss += loss.item()
            _, predict = torch.max(out, 1)
            correct_num = (predict == batch_y).sum()
            running_acc += correct_num.item()
            if i % 10 == 0:
                loss1 = loss.cpu()
                trainloss_count.append(loss1.detach().numpy())
                print('{}:\t'.format(i), loss.item())
                # torch.save(model, r'log_CNN1'+str(lrc)+str(epochs))
                # torch.save(model,'helpdeskmodel/helpdesk_CNN_new'+model.getstr()+str(ix)+'th')
        running_loss /= len(train_dataset)
        running_acc /= len(train_dataset)
        print("[%d/%d] Loss: %.5f, Acc: %.5f" % (epoch + 1, epochs, running_loss, 100 * running_acc))
        if epoch==15 and 100 * running_acc<89.5:
            break
    torch.save(model, 'helpdeskmodel/helpdesk_CNN_new' + model.getstr() + str(ix) + 'th')
    model.eval()
    testloss = 0
    testacc = 0
    outlist = []
    prelist = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            output1 = output.cpu()
            testloss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output1.max(1)[1]
            pre = pred.clone()
            pre = pre.detach().numpy()
            for i in pre:
                prelist.append(i)
            p = output1.max(1)[0]
            p = p.detach().numpy()
            for i in p:
                outlist.append(i)
            testacc += pred.eq(target).sum().item()
    testloss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(testloss, testacc,
            len(test_loader.dataset),100. * testacc / len(test_loader.dataset)))
    return 100. * testacc / len(test_loader.dataset)

def loadmodelandpre(test_loader,smodel,device):
    model=torch.load(smodel)
    model.eval()
    testloss = 0
    testacc = 0
    outlist = []
    prelist = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            output1 = output.cpu()
            testloss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output1.max(1)[1]
            pre = pred.clone()
            pre = pre.detach().numpy()
            for i in pre:
                prelist.append(i)
            p = output1.max(1)[0]
            p = p.detach().numpy()
            for i in p:
                outlist.append(i)
            testacc += pred.eq(target).sum().item()
    testloss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(testloss, testacc,
            len(test_loader.dataset),100. * testacc / len(test_loader.dataset)))
    return 100. * testacc / len(test_loader.dataset)

if __name__ == '__main__':
    start_time=time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = nptolist(npdata)
    datatrain, datatest = model_selection.train_test_split(npdata, train_size=0.7)
    datatrain = nptolist(datatrain)
    datatest = nptolist(datatest)
    eventlist, eventlistlen = geteventlist(data)
    print('eventlist:', len(eventlist))
    blanklist = initbkanktrace(eventlistlen)
    ll = getorder(data, eventlist, eventlistlen)
    lc = getconcurrencegraph(ll)
    ls = getqequencegraph(ll)
    le = getexclusivegraph(ll)
    datatraintrace, datatrainlabel,traindelindex = delevent(datatrain)
    datatesttrace, datatestlabel ,testdelindex= delevent(datatest)
    trainlabel1, trainlabel2 = [], []
    for i in range(len(datatrainlabel)):
        l = []
        l.append(datatrainlabel[i])
        trainlabel1.append(l)
    for i in range(len(datatestlabel)):
        l = []
        l.append(datatestlabel[i])
        trainlabel2.append(l)
    datatestlabel = labeltotorch_tensor(datatestlabel, eventlist)
    datatrainlabel = labeltotorch_tensor(datatrainlabel, eventlist)
    train_datal = np.array(datatrainlabel)
    test_datal = np.array(datatestlabel)
    multitrain1, multitrain2, multitrain3 = getmultigraph(datatraintrace, eventlist, lc, le, ls, blanklist,traindelindex)
    multitest1, multitest2, multitest3 = getmultigraph(datatesttrace, eventlist, lc, le, ls, blanklist,testdelindex)

    datatraintrace1 = pd.DataFrame(datatraintrace)
    datatesttrace2 = pd.DataFrame(datatesttrace)
    label1 = pd.DataFrame(trainlabel1)
    label2 = pd.DataFrame(trainlabel2)


    traindatas1 = torch.from_numpy(multitrain1)
    traindatas1 = traindatas1.float()
    traindatas2 = torch.from_numpy(multitrain2)
    traindatas2 = traindatas2.float()
    traindatas3 = torch.from_numpy(multitrain3)
    traindatas3 = traindatas3.float()

    testdatas1 = torch.from_numpy(multitest1)
    testdatas1 = testdatas1.float()
    testdatas2 = torch.from_numpy(multitest2)
    testdatas2 = testdatas2.float()
    testdatas3 = torch.from_numpy(multitest3)
    testdatas3 = testdatas3.float()

    trainlabeldatas = torch.from_numpy(train_datal)
    trainlabeldatas = trainlabeldatas.long()
    testlabeldatas = torch.from_numpy(test_datal)
    testlabeldatas = testlabeldatas.long()

    train_dataset1 = TensorDataset(traindatas1, trainlabeldatas)
    train_dataset2 = TensorDataset(traindatas2, trainlabeldatas)
    train_dataset3 = TensorDataset(traindatas3, trainlabeldatas)

    test_dataset1 = TensorDataset(testdatas1, testlabeldatas)
    test_dataset2 = TensorDataset(testdatas2, testlabeldatas)
    test_dataset3 = TensorDataset(testdatas3, testlabeldatas)

    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=50)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=50)
    train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=50)

    test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=100)
    test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=100)
    test_loader3 = torch.utils.data.DataLoader(dataset=test_dataset3, batch_size=100)

    loss_func = torch.nn.CrossEntropyLoss()
    acclist31, acclist32, acclist33 = [], [], []
    acclist21, acclist22, acclist23 = [], [], []
    acclist11, acclist12, acclist13 = [], [], []
    # for i in range(15):
    #     acclist31.append(trainandtest(train_loader1, test_loader1, CNN_Net1, loss_func, 0.003, train_dataset1, 50,device,i))
    #     acclist32.append(trainandtest(train_loader2, test_loader2, CNN_Net2, loss_func, 0.003, train_dataset1, 50,device,i))
    #     acclist33.append(trainandtest(train_loader3, test_loader3, CNN_Net3, loss_func, 0.003, train_dataset1, 50,device,i))
    # print(acclist31,acclist31.index(max(acclist31)),max(acclist31))
    # print(acclist32,acclist32.index(max(acclist32)),max(acclist32))
    # print(acclist33,acclist33.index(max(acclist33)),max(acclist33))

    pd1,prelist=loadmodelandpre(test_loader1,'helpdesk_CNN_new0.003CNN_Net1',device)
    pd2,_=loadmodelandpre(test_loader2,'helpdesk_CNN_new0.003CNN_Net2',device)
    pd3,_=loadmodelandpre(test_loader3,'helpdesk_CNN_new0.003CNN_Net3',device)
    print(pd1)
    print(pd2)
    print(pd3)
    end_time=time()
    print(end_time-start_time,":s"
                        )