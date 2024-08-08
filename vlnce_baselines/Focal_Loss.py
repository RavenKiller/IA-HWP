# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, reduce=True, size_average=True):
        """
        focal loss function
        :param alpha:   list (weights of every class) or scalar (the last class weight: alpha, others: 1-alpha)
        :param gamma:   
        :param num_classes:
        :param size_average:
        :param reduce:
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        if isinstance(alpha,list):
            assert len(alpha)==num_classes 
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[-1] += alpha
            self.alpha[:-1] += (1-alpha) # [1-α, 1-α, 1-α, 1-α, ..., α ] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))  
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(alpha, loss.t())
        if self.reduce:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
        
        return loss
