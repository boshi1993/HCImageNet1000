
import numpy as np
import torch
device=torch.device('cuda:0')
import os
from random import randint
import numpy as np

def b_distance(mx1,Sigma1,mx2,Sigma2,device=torch.device('cuda:0')):
    lamdba=0.01
    mu=mx1-mx2
    mu=torch.unsqueeze(mu, 1)
    Sigma=(Sigma2+Sigma1)/2
    dis= 1/8 *torch.transpose(mu,0,1)@torch.linalg.inv(Sigma)@mu
    dis=dis.squeeze()
    return dis
class PPCA:
    def __init__(self, model, nc=1000, d=640, num_pc=20, la=0.01,device=torch.device('cuda:0')):
        self.dimension = 200  # total number of Pricinple Component
        self.model = model
        self.d = d  # we need to know what is the dimension of features
        self.nc = nc  # number of classses
        self.device = torch.device('cuda:0')
        self.e = torch.eye(d, device='cpu')
        self.mu = torch.zeros((nc, d), dtype=torch.float, device='cpu')
        self.L = torch.zeros((nc, 200, d), dtype=torch.float, device='cpu')
        self.S = torch.zeros((nc, 200), dtype=torch.float, device='cpu')
        self.Sigma = torch.zeros((nc, d, d), dtype=torch.float, device='cpu')
        self.sxx = torch.zeros((nc, d, d), device='cpu',dtype=torch.float)
        # self.class_list = os.listdir(feature_path)  # this is a lest of number of classes,this is original ordered
        self.encode_list = []
        # self.n_pc = num_pc
        self.la = la
        self.disMatrix = torch.zeros((nc, nc),dtype=torch.float, device=device)

    def loadPara(self):
        para_path = 'D:/Experiment/HierarchicalC/ImageNet/parameters/{}/'.format(self.model)
        device = self.device
        nc = self.nc
        self.class_list = os.listdir(para_path)

        self.para_path = para_path
        i = 0
        for classes in self.class_list:
            full_feature_dir = '/'.join([self.para_path, classes])
            para_dict = torch.load(full_feature_dir)
            self.mu[i, :] = para_dict['mu']
            self.L[i, :, :] = para_dict['L']
            self.S[i, :] = para_dict['S']
            self.Sigma[i] = para_dict['Sigma']
            self.sxx[i] = para_dict['sxx']
            # self.S=torch.where(self.S>0.1,self.S,0.1*torch.ones(self.S.shape,device=device))
            i = i + 1

            class_name = "".join([classes.split(".")[0], '.pt'])
            self.encode_list.append(class_name)
        print(i)
        print('finish loading parameters')

    def generatePara(self):
        feature_path = 'D:/Experiment/IncrementalLearning/ImageNet/features/{}/train/'.format(self.model)
        device = self.device
        nc = self.nc
        self.class_list = os.listdir(feature_path)
        i = 0
        save_dir = 'D:/Experiment/HierarchicalC/ImageNet/parameters/{}/'.format(self.model)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        print('Start Generating parameters')
        for classes in self.class_list:
            full_feature_dir = ''.join([feature_path, classes])
            features = torch.load(full_feature_dir).to(device)
            class_name = classes.split(".")[0]
            n = features.shape[0]
            mu = torch.mean(features, dim=0)
            self.mu[i] = mu
            Sigma = torch.transpose((features - mu), 0, 1) @ (features - mu) / n

            # self.Sigma[i] = Sigma
            # sxx=(mu.unsqueeze(dim=1) @ mu.unsqueeze(dim=0) + Sigma)
            # self.sxx[i] = sxx

            u, s, v = torch.linalg.svd(Sigma)
            s[s < 0.01] = 0.01
            S = torch.diag(s)
            Sigma_new = u @ S @ torch.transpose(u, 0, 1)
            u, s, v = torch.linalg.svd(Sigma_new)
            self.L[i] = v[:200]
            self.S[i] = s[:200]
            self.encode_list.append(classes)
            self.Sigma[i] = Sigma_new
            sxx = (mu.unsqueeze(dim=1) @ mu.unsqueeze(dim=0) + Sigma)
            self.sxx[i] = sxx
            i = i + 1
            para_dict = {'class_name': classes, 'labels': i, 'mu': mu, 'L': v[0:200, :], 'S': s[0:200], 'Sigma': Sigma_new,
                         'sxx': sxx}
            save_dir1 = ''.join([save_dir, class_name, '.pth'])
            torch.save(para_dict, save_dir1)

    def computeDisMatrix(self):
        device = self.device
        print('Computing distance matrix')
        with torch.no_grad():
            nc = self.nc
            for i in range(1, nc):
                for j in range(i):
                    mu1 = self.mu[i].to(device)
                    Sigma1 = self.Sigma[i].to(device)
                    mu2 = self.mu[j].to(device)
                    Sigma2 = self.Sigma[j].to(device)
                    self.disMatrix[i, j] = b_distance(mu1, Sigma1, mu2, Sigma2, device=torch.device('cuda:0'))

        for i in range(0, nc - 1):
            for j in range(i + 1, nc):
                self.disMatrix[i, j] = self.disMatrix[j, i]
        dir = 'D:/Experiment/HierarchicalC/ImageNet/DistanceM/{}/'.format(self.model)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        dir1=dir+'bc_disMatrix.pth'
        torch.save(self.disMatrix, dir1)
        print('Finish printing distance matrix')

    def loadDisMatrix(self):
        dir = 'D:/Experiment/HierarchicalC/ImageNet/DistanceM/{}/bc_disMatrix.pth'.format(self.model)
        self.disMatrix = torch.load(dir)

    def selectInit(self, nsc):
        with torch.no_grad():
            disM = torch.where(self.disMatrix > 0, self.disMatrix, torch.zeros(self.disMatrix.shape, device=device))
            self.nsc = nsc
            self.init_list = torch.zeros(nsc, device=device).long()
            first_ind = randint(0, nsc - 1)
            self.init_list[0] = first_ind

            row = disM[first_ind]
            for i in range(1, self.nsc):
                ind = np.random.choice(self.nc, p=(row / sum(row)).cpu().numpy())
                self.init_list[i] = ind
                row, _ = torch.min(torch.stack((row, disM[ind]), dim=0), dim=0)


def KL(mu_q,invSigma_q,mu_p,Sigma_p,nsc=10,device=torch.device('cuda:0')):
    with torch.no_grad():
        n=mu_p.shape[0]
        mu_p=mu_p.to(device)
        Sigma_p=Sigma_p.to(device)
        mu=mu_p-mu_q
        score1=(mu.unsqueeze(1)@invSigma_q@mu.unsqueeze(2)).squeeze()
        score2=torch.sum(torch.diagonal(invSigma_q@Sigma_p, dim1=-1),dim=1)
        score3=torch.logdet(Sigma_p)+torch.logdet(invSigma_q)
    return score1+score2-score3

def generateSuper(mu,sigma):
    superMu=torch.mean(mu,dim=0)
    s_mu=mu-superMu
    n=mu.shape[0]
    d=mu.shape[1]
    xx=torch.reshape(s_mu,(n,d,1))@torch.reshape(s_mu,(n,1,d))
    superSigma=torch.mean(xx,dim=0)+torch.mean(sigma,dim=0)
    inverseSigma=torch.linalg.inv(superSigma)
    return superMu,superSigma,inverseSigma

class cluster:
    def __init__(self, model, nsc=10, nc=1000, d=640, n_pc=20, device=torch.device('cuda:0')):

        self.d = d
        self.nsc = nsc
        self.nc = nc
        self.mu = torch.zeros((nsc, d), device=device, dtype=torch.float)
        self.Sigma = torch.zeros((nsc, d, d), device=device, dtype=torch.float)
        self.sxx = torch.zeros((nsc, d, d), device=device, dtype=torch.float)
        self.invSigma = torch.zeros((nsc, d, d), device=device, dtype=torch.float)
        self.disMatrix = torch.zeros((nc, nsc), device=device, dtype=torch.float)
        self.ind = torch.zeros((nc, 0), device=device, dtype=torch.long).squeeze()
        self.L = torch.zeros((nsc, 200, d), device=device, dtype=torch.float)
        self.S = torch.zeros((nsc, 200), device=device, dtype=torch.float)
        self.indexDict = {}

    def getInitCent(self, mu, Sigma):
        self.Sigma = Sigma
        self.mu = mu
        with torch.no_grad():
            for i in range(self.nsc):
                self.invSigma[i] = torch.linalg.inv(self.Sigma[i])

    def getInitCent1(self, mu, sxx):
        self.sxx = sxx
        self.mu = mu
        with torch.no_grad():
            for i in range(self.nsc):
                self.Sigma[i] = self.sxx[i] - self.mu[i].unsqueeze(dim=1) @ self.mu[i].unsqueeze(dim=0)
                self.invSigma[i] = torch.linalg.inv(self.Sigma[i])

    def computeKL(self, mu_p, Sigma_p):
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(self.nsc):

                self.disMatrix[:,i] = KL(self.mu[i], self.invSigma[i], mu_p, Sigma_p,nsc= self.nsc)
            score, ind = torch.min(self.disMatrix, dim=1)
            self.ind = ind
            loss = torch.sum(score)
            print('loss is', loss)
        return loss
    """
    def updateParameter(self, mu_p, Sigma_p, sxx_p):
        # First we need update  parameters
        with torch.no_grad():
            for i in range(self.nsc):
                mu = mu_p[self.ind == i]
                sxx = sxx_p[self.ind == i]
                n = mu.shape[0]
                self.mu[i] = torch.mean(mu, dim=0)
                self.sxx[i] = torch.mean(sxx, dim=0)
                self.Sigma[i] = self.sxx[i] - self.mu[i].unsqueeze(dim=1) @ self.mu[i].unsqueeze(dim=0)
                self.invSigma[i] = torch.linalg.inv(self.Sigma[i])
    """
    def updateParameter(self, mu_p, Sigma_p, sxx_p):
        # First we need update  parameters
        for i in range(self.nsc):
            mu = mu_p[self.ind == i]
            Sigma = Sigma_p[self.ind == i]
            n = mu.shape[0]
            self.mu[i], self.Sigma[i], self.invSigma[i] = generateSuper(mu, Sigma)

    def extract(self):
        for i in range(self.nsc):
            u, s, v = torch.linalg.svd(self.Sigma[i], full_matrices=True)
            self.L[i] = v[0:200, :]
            self.S[i] = s[0:200]

    def getIndex(self):
        for i in range(self.nsc):
            self.indexDict[i] = []
        for i in range(self.nc):
            self.indexDict[int(self.ind[i])].append(i)

    def computeEnt(self):
        dt_v = torch.zeros(self.nsc)
        for i in range(self.nsc):
            dt_v[i] = torch.sum(self.ind == i)
            p_v = dt_v / self.nc
        ent = torch.sum(-torch.log2(p_v) * p_v)
        print('Entrop is', ent)
        self.ent = ent
        return ent