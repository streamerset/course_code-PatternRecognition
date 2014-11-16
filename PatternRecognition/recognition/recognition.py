# coding=gbk
#python edition: Python3.4.1,2014,10,18
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def read_points():
	dataset=[]
	with open('mydata.txt','r') as file:
		for line in file:
			if line =='\n':
				continue
			dataset.append(list(map(float,line.split(' '))))
		file.close()
		return  dataset

def generate_traineddata():
    arr=[[] for i in range(3)]
    with open('mybase.txt','r') as file:
        index=0
        for line in file:
            if line=='\n' :
                continue
            elif line[0]=='C':
                index=int(line[-2])-1
                continue
            arr[index].append(int(line))
        file.close()
    train=[[] for i in range(3)]
    test=[[] for i in range(3)]
    for i in range(len(arr)):
        tr=5
        train[i]=arr[i][:tr]
        test[i]=arr[i][tr:]
    return train,test

def createMatrix(train,test,dataset,feature):
    trainmat=[[] for i in range(3)]
    testmat=[[] for i in range(3)]
    extremum=[32767,32767,0,0]
    for i in range(3):
        for j in train[i]:
            datatr=[dataset[j][feature[0]],dataset[j][feature[1]]]
            trainmat[i].append(datatr)
            extremum[0],extremum[1]=min(dataset[j][feature[0]],extremum[0]),min(dataset[j][feature[1]],extremum[1])
            extremum[2],extremum[3]=max(dataset[j][feature[0]],extremum[2]),max(dataset[j][feature[1]],extremum[3])
        for k in test[i]:
            datate=[dataset[k][feature[0]],dataset[k][feature[1]]]
            testmat[i].append(datate)
            extremum[0],extremum[1]=min(dataset[k][feature[0]],extremum[0]),min(dataset[k][feature[1]],extremum[1])
            extremum[2],extremum[3]=max(dataset[k][feature[0]],extremum[2]),max(dataset[k][feature[1]],extremum[3])
    extremum[0],extremum[1]=int(extremum[0]) ,int(extremum[1])
    extremum[2],extremum[3]=int(extremum[2]+1) ,int(extremum[3]+1)
    return   trainmat,testmat,extremum

def createTrainmatAll(train,dataset):
    trainmatall=[]
    for i in range(3):
        for j in  train[i]:
            trainmatall.append(dataset[j])
    return trainmatall

def classify(trainmat,testmat,test):
    #求三类训练集的均值向量
    tr1,tr2,tr3=np.mat(trainmat[0]),np.mat(trainmat[1]),np.mat(trainmat[2])
    te=[[] for i in range(3)]
    te[0],te[1],te[2]=np.mat(testmat[0]),np.mat(testmat[1]),np.mat(testmat[2])
    M1=np.mean(tr1,axis=0)
    M2=np.mean(tr2,axis=0)
    M3=np.mean(tr3,axis=0)
    #获得矩阵长度
    l1,l2,l3=len(trainmat[0]),len(trainmat[1]),len(trainmat[2])
    #求协方差矩阵
    C1=np.cov(tr1.T)
    C2=np.cov(tr2.T)
    C3=np.cov(tr3.T)

    feature=open('my_feature.txt','a')
    print('每一类的均值向量：\n',  M1,'\n',M2,'\n',M3,end='\n',file=feature)
    print('每一类的协方差矩阵：\n',  C1,'\n',C2,'\n',C3,'\n',end='\n',file=feature)
    feature.close()
    result=[[] for i in range(4)]
    for i in range(3):
        testset=te[i]
        count=0
        for X in testset:
            r1=-0.5*np.log(np.fabs(la.det(C1)))-0.5*((X-M1))*(la.inv(C1))*((X-M1).T)
            r2=-0.5*np.log(np.fabs(la.det(C2)))-0.5*((X-M2))*(la.inv(C2))*((X-M2).T)
            r3=-0.5*np.log(np.fabs(la.det(C3)))-0.5*((X-M3))*(la.inv(C3))*((X-M3).T)
            if r1>r2 and r1>r3:
                result[0].append(test[i][count])
            elif  r2>r1 and r2>r3:
                result[1].append(test[i][count])
            elif  r3>r1 and r3>r2:
                result[2].append(test[i][count])
            else:
                result[3].append(test[i][count])
            count=count+1
    print(result)
    return result

def featureextract(trainmatall):
    tr=np.mat(trainmatall)
    M=np.mean(tr,axis=0)
    feature=open('my_feature.txt','w')
    print('训练集均值向量为：\n',M,end='\n',file=feature)
    C=np.cov(tr.T)
    print('协方差矩阵为：\n',C,end='\n',file=feature)
    w,v=la.eig(C)
    print('特征值为：\n',w,end='\n',file=feature)
    print('特征向量为：\n',v,end='\n',file=feature)
    feature.close()
    t1=[]
    for i in range(len(w)):
        t1.append([w[i],i])
    t2=sorted(t1,reverse=True)[:2]
    t3=[t2[0][-1],t2[1][-1]]
    print(t3)
    return t3

def plotfig(trainmat,testmat,resultmat,falsemat,extremum):
    fig=plt.figure("贝叶斯决策分类结果图",figsize=(16,12))
    plt.clf()
    subplot1=fig.add_subplot(1,3,1)
    lab=["First Kind","Second Kind","Third Kind"]
    clk=['k','r','b']
    mk=[(3,0),(4,2),(5,3)]
    for i in range(3):
        plt.scatter(0,0,color=clk[i],marker=mk[i],label=lab[i])
        for j in range(len(trainmat[i])):
               plt.scatter(trainmat[i][j][0],trainmat[i][j][1],color=clk[i],marker=mk[i])
    subplot1.set_xlabel('x')
    subplot1.set_ylabel('y')
    plt.xlim(extremum[0]-1,extremum[2])
    plt.ylim(extremum[1],extremum[3]+1)
    xt=[]
    for i in range(0,extremum[2]-extremum[0]+1,1):
        xt.append(extremum[0]+i)
    yt=[]
    for i in range(0,extremum[3]-extremum[1]+1,1):
       yt.append(extremum[1]+i)
    plt.xticks(xt)
    plt.yticks(yt)
    subplot1.set_title("sample distribution")
    subplot1.legend(loc='upper right')

    subplot2=fig.add_subplot(1,3,2)
    lab=["First Kind","Second Kind","Third Kind"]
    clk=['k','r','b']
    mk=[(3,0),(4,2),(5,3)]
    for i in range(3):
        plt.scatter(0,0,color=clk[i],marker=mk[i],label=lab[i])
        for j in range(len(testmat[i])):
               plt.scatter(testmat[i][j][0],testmat[i][j][1],color=clk[i],marker=mk[i])
    subplot2.set_xlabel('x')
    subplot2.set_ylabel('y')
    plt.xlim(extremum[0]-1,extremum[2])
    plt.ylim(extremum[1],extremum[3]+1)
    plt.xticks(xt)
    plt.yticks(yt)
    subplot2.set_title("test distribution")
    subplot2.legend(loc='upper right')

    subplot3=fig.add_subplot(1,3,3)
    lab=["First Kind","Second Kind","Third Kind","False Result"]
    clk=['k','r','b','g']
    mk=[(3,0),(4,2),(5,3),(4,0)]
    for i in range(3):
        plt.scatter(0,0,color=clk[i],marker=mk[i],label=lab[i])
        for j in range(len(resultmat[i])):
               plt.scatter(resultmat[i][j][0],resultmat[i][j][1],color=clk[i],marker=mk[i])
    plt.scatter(0,0,color=clk[-1],marker=mk[-1],label=lab[-1])
    for i in range(3):
        for j in range(len(falsemat[i])):
                       plt.scatter(falsemat[i][j][0],falsemat[i][j][1],color=clk[-1],marker=mk[-1])
    subplot3.set_xlabel('x')
    subplot3.set_ylabel('y')
    plt.xlim(extremum[0]-1,extremum[2])
    plt.ylim(extremum[1],extremum[3]+1)
    plt.xticks(xt)
    plt.yticks(yt)
    subplot3.set_title("bayes distribution")
    subplot3.legend(loc='upper right')
    plt.show()

def main():
    dataset=read_points()
    train,test=generate_traineddata()
    trainmatall=createTrainmatAll(train,dataset)
    feature=featureextract(trainmatall)
    trainmat,testmat,extremum=createMatrix(train,test,dataset,feature)
    result=classify(trainmat,testmat,test)
    falsemat=[[] for i in range(3)]
    resultmat=[[] for i in range(3)]
    for i in range(3):
        for j in result[i]:
            datare=[dataset[j][feature[0]],dataset[j][feature[1]]]
            if j not in test[i]:
                falsemat[i].append(datare)
            else:
                resultmat[i].append(datare)
    plotfig(trainmat,testmat,resultmat,falsemat,extremum)

if __name__=='__main__':
    main()