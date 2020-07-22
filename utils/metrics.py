
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class KNNClassification(nn.Module):

    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, weights='distance')
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)

        acc = accuracy_score(y_true, y_pred)

        return acc



class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        if instance_targets is not None:
            loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
            return prediction_logits, loss
        else:
            return prediction_logits



class NormSoftmaxLoss_W(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        self.weight_s1 = Parameter(torch.Tensor(dim//2, dim))
        self.weight_s2 = Parameter(torch.Tensor(num_instances, dim//2))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv_s1 = 1. / math.sqrt(self.weight_s1.size(1))
        stdv_s2 = 1. / math.sqrt(self.weight_s2.size(1))
        self.weight_s1.data.uniform_(-stdv_s1, stdv_s1)
        self.weight_s2.data.uniform_(-stdv_s2, stdv_s2)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        fc = nn.functional.linear(embeddings, self.weight_s1)
        fc = nn.functional.relu(fc)
        fc = nn.functional.linear(fc, self.weight_s2)
        w_sigma = torch.sigmoid(fc)

        if instance_targets is not None:
            loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
            return prediction_logits, loss
        else:
            return prediction_logits, w_sigma


class NormSoftmaxLoss_Margin(nn.Module):
    """ 
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """
    def __init__(self,
                 dim,
                 num_instances,
                 margin=0.5,
                 temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, label):
        norm_weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, norm_weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        loss = self.loss_fn(logits / self.temperature, label)
        return logits, loss




class HingeLoss(nn.Module):
    """
    Hinge loss based on the paper:
    when deep learning meets metric learning:remote sensing image scene classification
    via learning discriminative CNNs 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/9
    """

    def __init__(self, margin=0.44):
        super().__init__()
        
        self.margin = margin

    def forward(self, oneHotCodes, features):
        
        L_S = oneHotCodes.mm(torch.t(oneHotCodes))
        Dist = torch.norm(features[:,None] - features, dim=2, p=2)**2

        Dist = self.margin - Dist
        
        L_S[L_S==0] = -1

        Dist = 0.05 - L_S * Dist

        loss = torch.triu(Dist, diagonal=1)

        loss[loss < 0] = 0

        return torch.mean(loss)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()









