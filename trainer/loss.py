from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import Isomap


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0)
        losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)  # 2e-2

        return losses.mean()


def compute_hausdorff_distance(x, y):  # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance

    value1 = distance_matrix.min(1)[0].max(0)[0]
    value2 = distance_matrix.min(0)[0].max(0)[0]

    return torch.max(value1, value2)


def local_consistency_loss(anchor, positive, negative):
    distance_p_and_a = (anchor - positive).pow(2).sum(1)
    distance_n_and_a = (anchor - negative).pow(2).sum(1)
    loss = F.relu(distance_n_and_a - distance_p_and_a) + 1
    loss = loss.mean()
    return loss


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)  # 转置
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def geodesic_distance(x):
    '''
    Calculate the geodesic distance of any two points between matrices x and y
    @param x: N × m matrix
    @return: Geodesic distance matrix
    '''
    dtype = x.dtype
    x = x.detach().cpu().to(torch.double).numpy()
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    isomap.fit(x)
    dist_matrix = isomap.dist_matrix_

    return torch.from_numpy(dist_matrix).to(dtype)

def hausdorff_distance(x, y):
    distance_matrix_x2y = pairwise_distances(x, y)
    distance_matrix_y2x = pairwise_distances(y, x)
    value1 = torch.max(distance_matrix_x2y.min(1)[0], dim=0, keepdim=True)
    value2 = torch.max(distance_matrix_y2x.min(1)[0], dim=0, keepdim=True)
    value = torch.cat((value1[0], value2[0]))
    result = torch.max(value)
    return result


reconstruction_function = nn.MSELoss(reduction='sum')


def reconstruction_loss(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(AveragedHausdorffLoss, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.' \
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res
