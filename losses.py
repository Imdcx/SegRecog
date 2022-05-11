"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            ## select the image from different class
            ## mask [batch, batch]
            ## every row[i,j] represent if j-th image and i-th are same class 
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # default-->2
        contrast_count = features.shape[1]
        # [batch*2, feature_size]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            #[batch*2, 1]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            #[batch*2, feature_size]
            anchor_feature = contrast_feature
            # 2
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # [batch*2, batch*2]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        #  [batch*2, batch*2]
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # replace mask[batch*2,batch*2]'s diagonal elements to 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MMDConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                 sigma_list = [1, 2, 4, 8, 16], lamb = [-1, 2]):
        super(MMDConLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.sigma_list = sigma_list
        self.lamb = lamb

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #features_size = features.size(-1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            ## select the image from different class
            ## mask [batch, batch]
            ## every row[i,j] represent if j-th image and i-th are same class 
            mask = torch.eq(labels, labels.T).float().to(device)
            mask_ne = torch.ne(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # default-->2
        contrast_count = features.shape[1]
        # [batch*2, feature_size]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            #[batch*2, 1]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            #[batch*2, feature_size]
            anchor_feature = contrast_feature
            # 2
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # [batch*2, batch*2]
        #print(anchor_feature)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        # MMD Calculate
        diag_z = torch.diag(anchor_dot_contrast).unsqueeze(1)
        z_2 = diag_z.expand_as(anchor_dot_contrast)
        normal_2 = z_2 - 2 *anchor_dot_contrast + z_2.t()
        K = 0.0
        for sigma in self.sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += torch.exp(-gamma * normal_2)
        
        #print(anchor_dot_contrast)
        
        # tile mask
        #  [batch*2, batch*2]
        mask = mask.repeat(anchor_count, contrast_count)
        mask_ne = mask_ne.repeat(anchor_count, contrast_count)

        n_n = mask.sum(1, keepdim=True)
        n_m = n_n * mask_ne.sum(1, keepdim=True)
        
        mask = mask * self.lamb[0]
        mask_ne = mask_ne * self.lamb[1]
        # mask-out self-contrast cases
        # replace mask[batch*2,batch*2]'s diagonal elements to 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        K_XX = K * mask / torch.pow(n_n,2)
        K_XY = K * mask_ne / n_m
        loss = K_XX.sum(1) + K_XY.sum(1)
        loss = loss.sum()
        return loss


class DecayMMDConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                 sigma_list = [1, 2, 4, 8, 16], lamb = [1, 1], clamp=0.25, bound=[0.1,0.8]):
        super(DecayMMDConLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.sigma_list = sigma_list
        self.lamb = lamb
        self.clamp = 0.25
        self.bound = bound

    def forward(self, features, labels=None, mask=None, lamp=True):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #features_size = features.size(-1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            ## select the image from different class
            ## mask [batch, batch]
            ## every row[i,j] represent if j-th image and i-th are same class 
            mask = torch.eq(labels, labels.T).float().to(device)
            mask_ne = torch.ne(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # default-->2
        contrast_count = features.shape[1]
        # [batch*2, feature_size]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            #[batch*2, 1]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            #[batch*2, feature_size]
            anchor_feature = contrast_feature
            # 2
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        #  [batch*2, batch*2]
        mask = mask.repeat(anchor_count, contrast_count)
        mask_ne = mask_ne.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # replace mask[batch*2,batch*2]'s diagonal elements to 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # outer count
        n_n = mask.sum(1, keepdim=True)
        m_m = mask_ne.sum(1, keepdim=True)
        n_m = n_n * m_m
        # compute logits
        # [batch*2, batch*2]
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        
        # hinge p2
        mask_ne_pos = torch.ge(anchor_dot_contrast, self.bound[0]).float().to(device)
        mask_ne = mask_ne * mask_ne_pos

        mask_pos = torch.le(anchor_dot_contrast, self.bound[1]).float().to(device)
        mask = mask * mask_pos

        # print(mask.sum(1))

        # inner count
        # n_n = mask.sum(1, keepdim=True)
        # m_m = mask_ne.sum(1, keepdim=True)
        # n_m = n_n * m_m
        # print(n_n, m_m)

        # DecayMMD Calculate
        normal_2 = anchor_dot_contrast - 1
        mask = -mask
        normal_2 = (mask + mask_ne) * normal_2
        #### 
        K = 0.0
        for sigma in self.sigma_list:
            gamma = 1.0 / sigma
            K += torch.exp(gamma * normal_2)
        # count the number of positive and negative
        mask = -mask

        # set flag of K_XX and K_XY
        mask = mask * self.lamb[0]
        mask_ne = mask_ne * self.lamb[1]
        ### get MMD loss
        K_XX = K * mask / torch.pow(n_n,2)
        K_XY = K * mask_ne / n_m
        loss = K_XX.sum(1) + K_XY.sum(1)
        loss = loss.sum()
        return loss
###submit version
class InDecayMMDConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                 sigma_list = [1, 2, 4], lamb = [1, 1], clamp=0.25, bound=[0.1,0.8]):
        super(InDecayMMDConLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.sigma_list = sigma_list
        self.lamb = lamb
        self.clamp = 0.25
        self.bound = bound

    def forward(self, features, labels=None, mask=None, lamp=True):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #features_size = features.size(-1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            ## select the image from different class
            ## mask [batch, batch]
            ## every row[i,j] represent if j-th image and i-th are same class 
            mask = torch.eq(labels, labels.T).float().to(device)
            mask_ne = torch.ne(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # default-->2
        contrast_count = features.shape[1]
        # [batch*2, feature_size]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            #[batch*2, 1]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            #[batch*2, feature_size]
            anchor_feature = contrast_feature
            # 2
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        #  [batch*2, batch*2]
        mask = mask.repeat(anchor_count, contrast_count)
        mask_ne = mask_ne.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # replace mask[batch*2,batch*2]'s diagonal elements to 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # outer count
        n_n = mask.sum(1, keepdim=True)
        m_m = mask_ne.sum(1, keepdim=True)
        n_m = n_n * m_m
        # compute logits
        # [batch*2, batch*2]
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        
        # hinge p2
        mask_ne_pos = torch.ge(anchor_dot_contrast, self.bound[0]).float().to(device)
        mask_ne = mask_ne * mask_ne_pos

        mask_pos = torch.le(anchor_dot_contrast, self.bound[1]).float().to(device)
        mask = mask * mask_pos

        # print(mask.sum(1))

        # inner count
        # n_n = mask.sum(1, keepdim=True)
        # m_m = mask_ne.sum(1, keepdim=True)
        # n_m = n_n * m_m
        # print(n_n, m_m)

        # DecayMMD Calculate
        normal_2 = anchor_dot_contrast - 1
        # mask = -mask
        # normal_2 = (mask + mask_ne) * normal_2
        #### 
        K = 0.0
        for sigma in self.sigma_list:
            gamma = 1.0 / sigma
            K += torch.exp(gamma * normal_2)
        # count the number of positive and negative
        mask = -mask

        # set flag of K_XX and K_XY
        mask = mask * self.lamb[0]
        mask_ne = mask_ne * self.lamb[1]
        ### get MMD loss
        K_XX = K * mask / torch.pow(n_n,2)
        K_XY = K * mask_ne / n_m
        loss = K_XX.sum(1) + K_XY.sum(1)
        loss = loss.sum()
        return loss