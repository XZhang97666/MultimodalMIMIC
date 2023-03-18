import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import math
from torch.nn import BCELoss, CrossEntropyLoss
# Code adapted from the fairseq repo.
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          AdamW,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel
                         )
from module import *


def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""

    mask=mask.cpu().detach().numpy()
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for  i  in  range ( mask .shape[ 0 ]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


def recon_loss(x_ts, m1,m2,ypred,num_features):
    """ Autoencoder loss
    """
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    # wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
    #                3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
    # wc.shape = (1, num_features)
    y=x_ts.transpose(1,2)
    m1=m1.transpose(1,2)
    m2=m2.transpose(1,2)
    m2 = 1 - m2
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = torch.sum(m, dim=2)
    count = torch.where(count > 0, count,torch.ones_like(count))
    x = torch.sum(x, dim=2)/count
    # x = x/(wc**2)  # dividing by standard deviation
    x = torch.sum(x, dim=1)/num_features
    return torch.mean(x)


class S_Interp(nn.Module):
    def __init__( self,args,device,orig_d_ts):
        super(S_Interp, self).__init__()

        self.tt_max=args.tt_max
        self.device=device
        self.ref_t=torch.linspace(0, 1., self.tt_max).to(self.device)
        self.d_dim=orig_d_ts
        self.output = nn.Linear(args.embed_dim, args.embed_dim)
        self.kernel=  Parameter(torch.zeros(self.d_dim))

    def forward(self, x_ts, x_ts_mask, ts_tt_list,rec_mask,reconstruction=False):
        x_ts=x_ts.transpose(1,2)
        x_ts_mask=x_ts_mask.transpose(1,2)
        tt_len=ts_tt_list.shape[-1]
        d=ts_tt_list.unsqueeze(1)
        d=d.repeat(1,self.d_dim,1)
        if reconstruction:
            output_dim = tt_len
            m = rec_mask.transpose(1,2)
            ref_t=d.unsqueeze(-2).repeat(1,1,output_dim, 1)
            # ts_tt_list
            # ref_t = K.tile(d[:, :, None, :], (1, 1, output_dim, 1))

        else:
            m = x_ts_mask
            ref_t = self.ref_t.unsqueeze(0)
            output_dim = self.tt_max
        # import pdb; pdb.set_trace()
        d = d.unsqueeze(-1).repeat(1,1, 1,output_dim)
        mask =m.unsqueeze(-1).repeat(1,1, 1,output_dim)
        x_ts=x_ts.unsqueeze(-1).repeat(1,1,1,output_dim)


        norm = (d - ref_t)*(d - ref_t)
        a= torch.ones([self.d_dim,tt_len,output_dim]).to(self.device)

        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha =a*pos_kernel.unsqueeze(-1).unsqueeze(-1)
        w = torch.logsumexp(-alpha*norm + torch.log(mask+1e-12), dim=2)
        w1=w.unsqueeze(2).repeat(1,1,tt_len,1)
        w1 = torch.exp(-alpha*norm + torch.log(mask+1e-12)- w1)
        y = torch.sum(w1*x_ts, dim=2)
#
        w_t = torch.logsumexp(-10.0*alpha*norm + torch.log(mask+1e-12),
                          dim=2)  # kappa = 10
        w_t=w.unsqueeze(2).repeat(1,1,tt_len,1)

        w_t = torch.exp(-10.0*alpha*norm + torch.log(mask+1e-12) - w_t)
        y_trans = torch.sum(w_t*x_ts, dim=2)
        rep1 = torch.cat([y, w, y_trans], dim= 1)

        return rep1
class Cross_Interp(nn.Module):
    def __init__( self,args,device,orig_d_ts):
        super(Cross_Interp, self).__init__()
        self.device=device
        self.d_dim=orig_d_ts
        self.activation = nn.Sigmoid()
        self.cross_channel_interp =torch.empty(self.d_dim, self.d_dim).to(self.device)
        nn.init.eye_(self.cross_channel_interp)


    def forward(self, x,reconstruction=False):
        self.output_dim = x.shape[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :] #x
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2  =  w
        w=w.unsqueeze(-1).repeat(1,1,1,self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1)
        mean = mean.unsqueeze(1).repeat(1,self.output_dim,1)
        w2 = torch.matmul(w*(y - mean), cross_channel_interp) + mean
        rep1 = w2.permute(0, 2, 1)
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = torch.cat([rep1, intensity, y_trans], 1)
        return rep1


