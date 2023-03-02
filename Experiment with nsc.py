import torch
import argparse
import os
from util2 import PPCA,cluster
import time
import numpy as np
device=torch.device('cuda:0')

def train(cluster1,n_iter,exp,device=torch.device('cuda:0')):
    exp.selectInit(nsc=nsc)
    mu=exp.mu[exp.init_list].to(device)
    Sigma=exp.Sigma[exp.init_list].to(device)
    sxx=exp.sxx[exp.init_list].to(device)
    cluster1.getInitCent(mu,Sigma)
    for i in range(n_iter):

        loss=cluster1.computeKL(exp.mu,exp.Sigma)
        cluster1.getIndex()
        cluster1.updateParameter(exp.mu,exp.Sigma,exp.sxx)
    ent=cluster1.computeEnt()
    cluster1.extract()
    return loss,ent
    print('Finish training')

def response(x1, mu, L, S,d,q,la,device):
    # response using L,S (fast), using the math computation above
    # xSigma^-1 x = x^Tx/lambda - u^Tu/lambda where u = S/(S^2+lambda)^0.5Lx
    device =device
    x = x1 - mu
    Lx = x @ L.t()
    Qx = ( torch.sqrt(S) / torch.sqrt(S  + la)).unsqueeze(0) * Lx
    return (torch.sum(x ** 2, dim=1) - torch.sum(Qx ** 2, dim=1)) / la+(d-q)*np.log(la)+sum(torch.log(S+la))

def classify_super(x, mx, L, S,d,q,la,ns,device):
    device = device
    nj = x.shape[0]

    nc = mx.shape[0]
    out = torch.zeros(nj, nc, device=device)
    for i in range(nc):
        r = response(x, mx[i, :], L[i, :q, :], S[i, :q],d,q,la,device)
        out[:, i] = r
        sorted, indices = torch.sort(out,dim=1)
    return indices[:,:ns]

def classify_ori(x, mx, L, S, supermask, d, q, device=torch.device('cuda:0'),la=0.01):
    """

    :param x:
    :param mx:
    :param L:
    :param S:
    :param supermask:
    :param d:
    :param q:
    :param device:
    :param la:
    :return:
    """
    device = device
    nj = x.shape[0]

    nc = mx.shape[0]
    out = 10000 * torch.ones(nj, nc, device=device)
    for i in range(nc):
        r = response(x[torch.gt(supermask[:, i], 0)], mx[i, :], L[i, :q, :], S[i, :q], d, q, la,device=device)
        out[torch.gt(supermask[:, i], 0), i] = r

    py = torch.argmin(out, dim=1)
    return py
"""


"""
def test(cluster1, test_loader1,exp,ns,q,device):
    total = 0
    wrong = 0
    device = device
    total_wrong = 0
    super_wrong=0
    start = time.time()
    density = 0
    with torch.no_grad():
        for data in test_loader1:
            features, labels = data
            labels = torch.squeeze(labels)
            features, labels = features.to(device), labels.to(device)
            py = classify_super(features, cluster1.mu, cluster1.L, cluster1.S, d=d, q=q,  la=0.01, ns=ns,device=device)
            superlabels=cluster1.ind[labels.long()]
            super_wrong += torch.sum(superlabels.unsqueeze(dim=1) == py)
            supermask = torch.zeros(len(py), nc).to(device)
            i = 0
            for i in range(len(py)):
                for j in range(ns):
                    superlabel = py[i][j]
                    superIndex = cluster1.indexDict[int(superlabel)]

                    supermask[i][superIndex] = 1

            density += supermask.sum()
            label = classify_ori(features, exp.mu.to(device), exp.L.to(device), exp.S.to(device), supermask, d, q, la=0.01,device=device)

            total_wrong += torch.sum(label != labels)
            # wrong_list1.append(wrong/len(py))

            total += len(py)
        end = time.time()

    print('total error is',total_wrong / total)
    print('desity of supermaks :', density / (total * nc))
    print('Computation time is', end - start)
    result={'density':density / (total * nc),'error':total_wrong / total,'super_error':super_wrong/total}
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='Clip',help='which model do we use to extract features')
    parser.add_argument('-computeD', type=bool, default=False)
    parser.add_argument('-loadPara', type=bool, default=True)
    parser.add_argument('-d', type=int, default='640', help='What dimension of the features')
    parser.add_argument('-nc', type=int, default='1000', help='how many classes are in the dataset ')
    parser.add_argument('-n_iter', type=int, default='8  ', help='how many iteration for training ')
    parser.add_argument('-q', type=int, default='20', help='number of pc in classification ')
    parser.add_argument('-ns', type=int, default='4', help='number of suoercluster we use in classification ')
    parser.add_argument('-n_runs', type=int, default='4', help='number of runs of experiment')
"""
The objective of this project train and test hierarchical models for classification tasks.
The first stage is training it contained the initilization with k means++ and training with k means
The second stage is the preparation of dataset and inference


"""
    args = parser.parse_args()
    d = args.d
    nc = args.nc
    model = args.model
#prepare the models from predefined class
    PPCA1=PPCA(model=model,d=d,nc=nc,device=device)
    PPCA1.loadPara() #load the parameter for output class
    if args.computeD: #whether load the distance matrix
        PPCA1.computeDisMatrix()
    else:
        PPCA1.loadDisMatrix()

#prepare the traininig dataset for features to train
    feature_tensor = torch.zeros([0, d], dtype=torch.float32)
    labels = torch.zeros(0)
    for j in range(nc):
        classes = PPCA1.encode_list[j]
        x = torch.load(
            os.path.join('D:/Experiment/IncrementalLearning/ImageNet/features/{}/val/{}'.format(model, classes)))
        # x=x.to(device)
        # py=superClassify(x,super1.mu,super1.inverseSigma,n=3)
        label = j * torch.ones([x.shape[0]])
        feature_tensor = torch.cat([feature_tensor, x], 0)
        labels = torch.cat([labels, label], 0)
    test_data1 = torch.utils.data.TensorDataset(feature_tensor, labels)
    test_loader1 = torch.utils.data.DataLoader(
        test_data1,
        batch_size=1000,
        shuffle=False,
        num_workers=4
    )
    #nsc=10
    torch.cuda.empty_cache()#make sure we have enough gpu memory
    n_iter=args.n_iter #number of iteration for training
    n_s=args.ns
    #q_list=[5,10,20,50,100]
    #cluster1 = cluster(model=model, d=d, nsc=nsc, nc=nc)
    #loss, ent = train(cluster1, n_iter, PPCA1,device=device)
    #result = test(cluster1, test_loader1, PPCA1, ns=ns,q=q,device=device)
    #print('finish testing')
    n_runs=args.n_runs# number of independent training experiment
    nsc_list=[10,20,33,40,50,66,100]# number of super classes we want to try
    #matrix we use to record the results
    ent_m=torch.zeros([n_runs,len(nsc_list)],device='cpu')
    density_m=torch.zeros([n_runs,len(nsc_list),n_s],device='cpu')
    error_m = torch.zeros([n_runs, len(nsc_list),n_s],device='cpu')
    error_super_m = torch.zeros([n_runs,len(nsc_list),n_s,],device='cpu')
    q=20
    for i in range(n_runs):
        print('{}th run'.format(i))
        j = 0
        for nsc in nsc_list:

            print('Number of super clusters',nsc)
            torch.cuda.empty_cache()
            cluster1 = cluster(model=model, d=d, nsc=nsc, nc=nc)
            loss, ent = train(cluster1, n_iter, PPCA1, device=device)
            ent_m[i,j]=ent
            for ns in range(1,n_s+1):
                print('number of super:', ns)
                result = test(cluster1, test_loader1, PPCA1, ns=ns,q=q,device=device)
                density_m[i,j,ns-1]=result['density']
                error_m[i,j,ns-1]=result['error']
                error_super_m[i,j,ns-1]=result['super_error']
            j=j+1

    result_all={}
    result_all['ent']=ent_m
    result_all['error']=error_m
    result_all['density']=density_m
    result_all['error_super']=error_super_m
    save_dir = 'D:/Experiment/HierarchicalC/ImageNet/result/{}/'.format(args.model)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dir1 = save_dir + 'nsc_result.pth'
    torch.save(result_all, dir1)
