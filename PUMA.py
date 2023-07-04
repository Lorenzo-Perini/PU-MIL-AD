from __future__ import division
from __future__ import print_function

import torch
#from torch import nn
#from multiprocessing import Pool, freeze_support, cpu_count, set_start_method

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from pyod.models.base import BaseDetector
from pyod.utils.torch_utility import get_activation_by_name
from pyod.utils.stat_models import pairwise_distances_no_broadcast
import io
from torch.utils.tensorboard import SummaryWriter

class PyODDataset(torch.utils.data.Dataset):

    def __init__(self, X_bags, y=None, mean=np.zeros(1,np.float), std=np.ones(1,np.float)):
        super(PyODDataset, self).__init__()
        self.n_bags = X_bags.shape[0]
        self.n_samples = X_bags.shape[1]
        self.n_features = X_bags.shape[2]
        self.X_bags = X_bags
        self.X_inst = X_bags.reshape(self.n_bags*self.n_samples,self.n_features)
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X_bags.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X_bags[idx, :, :]
        sample = (sample - self.mean) / self.std

        return torch.from_numpy(sample), idx


class inner_autoencoder(torch.nn.Module):
    def __init__(self,
                 n_features,
                 hidden_neurons=[128, 64],
                 dropout_rate=0.2,
                 batch_norm=True,
                 hidden_activation='relu'):
        super(inner_autoencoder, self).__init__()
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.hidden_activation = hidden_activation

        self.activation = get_activation_by_name(hidden_activation)

        self.layers_neurons_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.encoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_[idx]))
            self.encoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_[idx],self.layers_neurons_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.encoder.add_module("dropout"+str(idx),torch.nn.Dropout(dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if batch_norm:
                self.decoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                                      self.layers_neurons_decoder_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.decoder.add_module("dropout"+str(idx),torch.nn.Dropout(dropout_rate))

    def forward(self, x):
        # we could return the latent representation here after the encoder as the latent representation
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class PUMA(BaseDetector):

    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 batch_norm=True,
                 learning_rate=0.01,
                 epochs=100,
                 batch_size=10,
                 mu1 = 0, 
                 sigma1 = 0.1,
                 mu2 = 1,
                 sigma2 = 0.1,
                 n_neg = 10,
                 dropout_rate=0.1,
                 weight_decay=1e-5,
                 preprocessing=True,
                 contamination=0.1,
                 random_state = 331,
                 gpu = -1,
                 verbose = True,
                 cont_factor = 0.1):
        torch.manual_seed(random_state)
        super(PUMA, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.n_neg = n_neg
        self.loss_fn = torch.nn.MSELoss()
        #self.writer = SummaryWriter('/home/lorenzo/projects/MI_Learning/csvfiles/log/')
        self.cont_factor = cont_factor
        
        if gpu == -1:
            self.device = torch.device("cpu")
        elif gpu in [0,1,2]:
            self.device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        self.verbose = verbose
        
    # noinspection PyUnresolvedReferences
    def fit(self, X_bags, Y_bags):

        # validate inputs X and y (optional)
        #X_inst = check_array(X_inst)
        self.n_bags = X_bags.shape[0]
        self.n_samples = X_bags.shape[1]
        self.n_features = X_bags.shape[2]
        
        if self.n_neg == -1:
            npos = len(np.where(Y_bags == 1)[0])
            self.n_neg = npos
            
        if self.cont_factor>0:
            tot_inst = self.n_bags*self.n_samples
            inst_labeledbags = len(np.where(Y_bags == 1)[0]) *self.n_samples
            inst_unlabeledbags = (self.n_bags - len(np.where(Y_bags == 1)[0])) *self.n_samples
            self.cont_factor = max((self.cont_factor*tot_inst - 0.25*inst_labeledbags)/inst_unlabeledbags,0)
        Y_bags = torch.from_numpy(Y_bags)
        X_inst = X_bags.reshape(self.n_bags*self.n_samples,self.n_features)
        
        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X_inst, axis=0), np.std(X_inst, axis=0)
            train_set = PyODDataset(X_bags=X_bags, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X_bags=X_bags)
        
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers = 0,
                                                   pin_memory = False)
        
        # initialize the model
        self.model = inner_autoencoder(
            n_features=self.n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)
        
        self.model.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.model.A.grad = torch.tensor(torch.rand(1))
        self.model.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        
        # move to device and print model information
        self.model = self.model.to(self.device, non_blocking = True)
        if self.verbose:
            print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader, Y_bags)

        self.model.load_state_dict(self.best_model_dict)
        self.bag_decision_scores_, self.instance_decision_scores_ = self.decision_function(X_bags)
        self._process_decision_scores()
        return self


    def _train_autoencoder(self, train_loader, Y_bags):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay, amsgrad = True)
        self.best_loss = float('inf')
        self.best_model_dict = None
        y_tmp = torch.clone(Y_bags)
        neg_idx = torch.where(Y_bags == 0)[0]
        y_tmp[neg_idx[torch.randperm(neg_idx.size(0))[:self.n_neg]]] = -1
            
        #
        for epoch in range(self.epochs):
            overall_loss = []
            loss1 = []
            loss2 = []
            bag_scores = torch.zeros([self.n_bags, 1]).to(self.device, non_blocking = True).float()
            for data, data_idx in train_loader:
                idx_l1 = torch.where(y_tmp[data_idx] != 1)[0]
                local_batch_size,_,_ = data.size()
                optimizer.zero_grad()
                data_inst = torch.reshape(data, (local_batch_size*self.n_samples, self.n_features))
                data_inst = data_inst.to(self.device, non_blocking = True).float()
                if idx_l1.shape[0] >0:
                    data_inst_l1 = torch.reshape(data[idx_l1,:,:],(len(idx_l1)*self.n_samples, self.n_features))
                    data_inst_l1 = data_inst_l1.to(self.device, non_blocking = True).float()
                    if self.cont_factor>0:
                        l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1,self.model(data_inst_l1))
                        data_inst_l1 = data_inst_l1[torch.where(l1<torch.quantile(l1, 1-self.cont_factor, dim=0))]
                    loss = self.loss_fn(data_inst_l1, self.model(data_inst_l1))
                else:
                    loss = torch.tensor(0, dtype = torch.float).to(self.device, non_blocking = True).float()
                    
                loss1.append(loss.item())
                l1 = torch.nn.PairwiseDistance(p=2)(data_inst,self.model(data_inst)).reshape(local_batch_size, #unsqueeze
                                                                                             self.n_samples, 1)
                l2 = self._ss_loss(l1, y_tmp[data_idx])
                loss2.append(l2.item())
                #print("Loss1:", loss)
                #print("Loss2:", l2)
                loss += l2
                #print("L2:",l2)
                loss.backward()
                #print("Model A:", self.model.A, self.model.A.grad)
                #print(sum([torch.isfinite(x).all() for x in self.model.parameters()]))
                #print(sum([torch.isfinite(x.grad).all() for x in self.model.parameters()]))

                #torch.nn.utils.clip_grad_norm_(self.model.A, max_norm = 100, error_if_nonfinite = True)
                #print([x.grad for x in self.model.parameters()])
                optimizer.step()
                overall_loss.append(loss.item())
                
                #Need the following to compute the most reliable negative
                instance_scr=self._logistic(l1)
                pi = self._weightnoisyor(instance_scr)
                bag_scores[data_idx] = pi.reshape(local_batch_size,1)
                
            # Get the most reliable negatives:
            nonpos_idx = torch.where(Y_bags == 0)[0]
            sorted_idx = torch.argsort(bag_scores[nonpos_idx], dim=0)[:self.n_neg]
            y_tmp = torch.clone(Y_bags)
            y_tmp[nonpos_idx[sorted_idx]] = -1
            
            if self.verbose:
                print('epoch {epoch}: training loss {l} = (Lu {l1} + Lp {l2})'.format(epoch=epoch, l=np.mean(overall_loss),
                                                                                      l1=np.mean(loss1), l2=np.mean(loss2)))
            if np.mean(overall_loss) <= self.best_loss:
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()
    
    def _ss_loss(self, l1, y_bags):
        loss_index = torch.where(y_bags != 0)[0]
        if loss_index.nelement()>0:
            l1 = l1[loss_index,:,:]
            y = y_bags[loss_index]
            pij = self._logistic(l1)
            pi = self._weightnoisyor(pij)
            likelihood = -1*(self._log_diverse_density(pi, y)+1e-10) + 0.01*(self.model.A**2+self.model.B**2)[0]
        else:
            likelihood =  0.01*(self.model.A**2+self.model.B**2)[0]
        return likelihood
    
    def _logistic(self,loss):
        return torch.sigmoid(self.model.A*loss+self.model.B)
    
    def _noisy_or(self,pij):
        # instance_prob contains the probability of being positive
        noisyor = 1 - torch.prod(1-pij, dim = 1)
        return noisyor


    def _weightnoisyor(self,pij):
        rv1 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu1), scale=torch.tensor(self.sigma1))
        rv2 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu2), scale=torch.tensor(self.sigma2))
        nbags = pij.size()[0]
        ninstances = pij.size()[1]
        pij = pij.reshape(nbags,ninstances)
        ranks = torch.empty((nbags, ninstances), dtype = torch.float)
        tmp = torch.argsort(pij, dim=1, descending=False)
        for i in range(nbags):
            ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
        w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
        w = torch.div(w,torch.sum(w, dim = 1).reshape(nbags,1))
        pij = pij.to(self.device, non_blocking = True).float()
        w = w.to(self.device, non_blocking = True).float()
        noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min = 0, max = 1), dim = 1)
        return noisyor
    
    def _log_diverse_density(self, pi, y_bags):
        # Compute the likelihood given bag labels y_bags and bag probabilities pi
        z = torch.where(y_bags == -1)[0]
        if z.nelement() > 0:
            zero_sum = torch.sum(torch.log(1-pi[z]+1e-10))
        else:
            zero_sum = torch.tensor(0).to(self.device, non_blocking = True).float()
            
        o = torch.where(y_bags == 1)[0]
        if o.nelement() > 0:
            one_sum = torch.sum(torch.log(pi[o]+1e-10))
        else:
            one_sum = torch.tensor(0).to(self.device, non_blocking = True).float()
        return zero_sum+one_sum

    def decision_function(self, X_bags):

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X_bags=X_bags, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X_bags=X_bags)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()
        
        # construct the vector for holding the reconstruction error
        b_scores = torch.zeros([X_bags.shape[0], 1]).to(self.device, non_blocking = True).float()
        i_scores = torch.zeros([X_bags.shape[0]*X_bags.shape[1], 1]).to(self.device, non_blocking = True).float()

        with torch.no_grad():
            for data, data_idx in dataloader:
                local_batch_size, _, _ = data.size()
                data_idx = data_idx.to(self.device, non_blocking = True)
                mi = data_idx[0]
                ma = data_idx[local_batch_size-1]+1
                data_inst = torch.reshape(data, (local_batch_size*self.n_samples, self.n_features))
                data_inst = data_inst.to(self.device, non_blocking = True).float()
                l1 = torch.nn.PairwiseDistance(p=2, eps=0)(data_inst,self.model(data_inst))
                l1 = torch.reshape(l1, (local_batch_size, self.n_samples, 1))
                instance_scr = self._logistic(l1)
                i_scores[mi*self.n_samples:ma*self.n_samples]=instance_scr.reshape(local_batch_size*self.n_samples,1)
                pi = self._weightnoisyor(instance_scr)
                b_scores[data_idx] = pi.reshape(local_batch_size,1)

        return b_scores.cpu().numpy(), i_scores.cpu().numpy()
    
    def _process_decision_scores(self):

        self.threshold_ = 0.5
        self.bag_labels_ = (self.bag_decision_scores_ > self.threshold_).astype('int').ravel()
        self.instance_labels_ = (self.instance_decision_scores_ > self.threshold_).astype('int').ravel()

        return self
    
