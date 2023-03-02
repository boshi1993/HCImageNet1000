import torch
import argparse
import os
from util1 import RBF,cluster
import time

device=torch.device('cuda:0')


def train(cluster1, n_iter, RBF1, device=device):
    RBF1.selectInit(nsc=nsc)
    mu = RBF1.mu[RBF1.init_list].to(device)
    cluster1.getInitCent(mu)
    for i in range(n_iter):
        loss = cluster1.computeD(RBF1.mu)
        cluster1.getIndex()
        cluster1.updateParameter(RBF1.mu)
    ent = cluster1.computeEnt()
    print('Finish training')
    return ent

def classify_super(features,mu,ns,device):
    device = device
    out = torch.cdist(features.to(device),mu, p=2)#out is a b by nsc matrix
    score,ind=torch.sort(out,dim=1)

    return ind[:,:ns]
def classify(features,mu,supermask,device):
    device = device
    out = torch.cdist(features.to(device), mu, p=2)  # out is a b by nsc matrix

    out1=out*supermask

    score, ind = torch.min(out1, dim=1)


    return ind

def test(cluster1, test_loader1, RBF1, ns,nc,device=device):
    total = 0
    wrong = 0
    device = device
    total_wrong = 0
    super_wrong = 0
    start = time.time()
    density = 0
    with torch.no_grad():
        for data in test_loader1:
            features, labels = data
            labels = torch.squeeze(labels)
            features, labels = features.to(device), labels.to(device)
            py = classify_super(features, cluster1.mu,ns=ns,device=device)
            superlabels = cluster1.ind[labels.long()]
            super_wrong += torch.sum(superlabels.unsqueeze(dim=1) == py)
            supermask = torch.ones(len(py), nc).to(device)*10000
            i = 0
            for i in range(len(py)):
                for j in range(ns):
                    superlabel = py[i][j]
                    superIndex = cluster1.indexDict[int(superlabel)]

                    supermask[i][superIndex] = 1


            label = classify(features, RBF1.mu.to(device), supermask, device=device)
            #supermask=torch.where(supermask == 1, supermask, 0)
            density += (supermask==1).sum()

            total_wrong += torch.sum(label != labels)
            # wrong_list1.append(wrong/len(py))

            total += len(py)
        end = time.time()

    print('total error',total_wrong / total)
    print('desity of supermaks :', density / (total * nc))
    print('Computation time is', end - start)
    print ('Supererror is ', super_wrong / total)
    result = {'density': density / (total * nc), 'error': total_wrong / total, 'super_error': super_wrong / total}
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

    args = parser.parse_args()

    nsc_range=[10,20,33,40,50,66,100]
    d = args.d
    nc = args.nc
    model = args.model
    RBF1=RBF(model=model,d=d,nc=nc)
    #RBF1.generatePara()
    RBF1.loadPara()
    RBF1.computeDisMatrix()

    feature_tensor = torch.zeros([0, d], dtype=torch.float32)
    labels = torch.zeros(0)
    for j in range(nc):
        classes = RBF1.encode_list[j]
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
    nsc = 20
    n_runs = args.n_runs
    torch.cuda.empty_cache()
    n_iter = args.n_iter
    n_super = args.ns  # the maximum number of supercluster we use in inference
    q = args.q

    density_m = torch.zeros([n_runs, len(nsc_range), n_super], device='cpu')
    error_m = torch.zeros([n_runs, len(nsc_range), n_super], device='cpu')
    error_super_m = torch.zeros([n_runs, len(nsc_range), n_super], device='cpu')
    ent_m = torch.zeros([n_runs, len(nsc_range)], device='cpu')
    j = 0




    for nsc in nsc_range:
        print('number of superclusters:', nsc)
        for i in range(n_runs):
            print('{}th run'.format(i))
            torch.cuda.empty_cache()
            cluster1 = cluster(model=model,d=d,nsc=nsc,nc=nc)
            ent = train(cluster1, n_iter, RBF1, device)
            ent_m[i, j] = ent

            for ns in range(1, n_super + 1):
                print('number of super:', ns)
                result= test(cluster1, test_loader1, RBF1, ns,nc,device=device)
                density_m[i, j, ns - 1] = result['density']
                error_m[i, j, ns - 1] = result['error']
                error_super_m[i, j, ns - 1] = result['super_error']
        j = j + 1
    result_all = {}
    result_all['ent'] = ent_m
    result_all['error'] = error_m
    result_all['density'] = density_m
    result_all['error_super'] = error_super_m
    save_dir = 'D:/Experiment/HierarchicalC/ImageNet/result/{}/'.format(args.model)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dir1 = save_dir + 'RBF_result.pth'
    torch.save(result_all, dir1)