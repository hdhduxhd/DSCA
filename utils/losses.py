import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)],
                             dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)

def _KD_loss(pred, soft, T=2):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

class AugmentedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, norm=2, soft_margin=False):
        super(AugmentedTripletLoss, self).__init__()
        self.margin = margin
        self.norm = norm
        self.soft_margin = soft_margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, center=[]):
        device = inputs.device
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        num_proto = len(center)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if dist[i][mask[i] == 0].numel() == 0:
                dist_an.append((dist[i][mask[i]].max()+self.margin).unsqueeze(0))
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        if num_proto > 0:
            center = torch.from_numpy(center / np.linalg.norm(center, axis=1)[:, None]).to(device)
            for i in range(n):
                for j in range(num_proto):
                    distp = torch.norm(inputs[i].unsqueeze(0) - center[j], self.norm).clamp(min=1e-12)
                    dist_an[i] = min(dist_an[i].squeeze(0), distp).unsqueeze(0)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.soft_margin:
            loss = F.soft_margin_loss(dist_ap - dist_an, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class ContrastiveReplayLoss(nn.Module):
    def __init__(self, temperature=0.07, n_samples_per_class=5, current_weight=1.0, old_weight=0.8):
        super().__init__()
        self.temperature = temperature
        self.n_samples_per_class = n_samples_per_class
        self.current_weight = current_weight
        self.old_weight = old_weight

    def forward(self, features, targets, class_means=None, class_covs=None, known_classes=0, device="cuda:0"):
        batch_size = features.size(0)

        # ===== Step 1. 从旧类分布采样 =====
        old_samples_list, old_labels_list = [], []
        if class_means is not None and class_covs is not None:
            assert len(class_means) == len(class_covs) == known_classes
            for class_idx in range(known_classes):
                cls_mean = torch.tensor(class_means[class_idx], dtype=torch.float32).to(device)
                cls_cov = class_covs[class_idx].to(device)

                try:
                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                except:
                    cls_cov = cls_cov + torch.eye(cls_mean.shape[-1], device=cls_cov.device) * 1e-3
                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                samples = m.sample((self.n_samples_per_class,))
                old_samples_list.append(samples)
                old_labels_list.append(torch.full((self.n_samples_per_class,), class_idx, device=device, dtype=torch.long))

        if old_samples_list:
            old_samples = torch.cat(old_samples_list, dim=0)
            old_labels = torch.cat(old_labels_list, dim=0)
        else:
            old_samples = torch.empty(0, features.size(1), device=device)
            old_labels = torch.empty(0, device=device, dtype=torch.long)

        # ===== Step 2. 合并特征 =====
        combined_features = torch.cat([features, old_samples], dim=0)
        combined_labels = torch.cat([targets + known_classes, old_labels], dim=0)
        n_total = combined_features.size(0)

        # ===== Step 3. 相似度矩阵 =====
        norm_feats = F.normalize(combined_features, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_feats, norm_feats.T) / self.temperature

        # 正样本 mask
        label_matrix = combined_labels.unsqueeze(0) == combined_labels.unsqueeze(1)
        eye_mask = torch.eye(n_total, dtype=torch.bool, device=device)
        pos_mask = label_matrix & ~eye_mask

        # ===== Step 4. InfoNCE loss =====
        numerator = (similarity_matrix * pos_mask.float()).sum(dim=1, keepdim=True)
        numerator = numerator / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

        denominator = torch.logsumexp(
            similarity_matrix.masked_fill(eye_mask, float('-inf')), dim=1, keepdim=True
        )

        contrastive_loss = -(numerator - denominator)
        loss_current, loss_old = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # 平衡新旧样本
        if old_samples.numel() > 0:
            loss_current = contrastive_loss[:batch_size].mean()
            loss_old = contrastive_loss[batch_size:].mean()
            contrastive_loss = self.current_weight * loss_current + self.old_weight * loss_old
        else:
            contrastive_loss = contrastive_loss.mean()

        return contrastive_loss, loss_current, loss_old
    


class CenterAggregationLoss(nn.Module):
    def __init__(self, num_classes=200, feat_dim=768, margin=0.5, alpha=0.5):
        """
        中心聚合损失
        
        Args:
            num_classes: 类别数量
            feat_dim: 特征维度
            margin: 类内距离的目标边界
            alpha: 损失权重系数
        """
        super(CenterAggregationLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.alpha = alpha
        
        # 可学习的类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, labels):
        """
        Args:
            features: 特征张量 [batch_size, feat_dim]
            labels: 标签 [batch_size]
        """
        batch_size = features.size(0)
        
        # 计算每个特征到其对应类别中心的距离
        centers_batch = self.centers[labels]  # [batch_size, feat_dim]
        
        # 计算特征到类别中心的距离（使用L2距离）
        distances = torch.norm(features - centers_batch, p=2, dim=1)  # [batch_size]
        
        # 聚合损失：鼓励特征靠近其类别中心
        aggregation_loss = torch.mean(torch.clamp(distances - self.margin, min=0)**2)
        
        # 可选：添加中心更新机制（移动平均）
        # with torch.no_grad():
        #     for label in torch.unique(labels):
        #         class_mask = (labels == label)
        #         if class_mask.sum() > 0:
        #             class_features = features[class_mask]
        #             # 移动平均更新中心
        #             self.centers.data[label] = 0.9 * self.centers.data[label] + 0.1 * class_features.mean(dim=0)
        
        return self.alpha * aggregation_loss
    

class ContrastiveClusteringLoss(nn.Module):
    def __init__(self, margin=2.0, temperature=0.1, alpha=1.0, beta=0.5):
        """
        对比聚类损失：结合类内聚合和类间分离
        
        Args:
            margin: 类间分离的边界
            temperature: 对比损失的温度参数
            alpha: 类内聚合损失权重
            beta: 类间分离损失权重
        """
        super(ContrastiveClusteringLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        # 计算特征相似度矩阵
        similarity_matrix = torch.matmul(features, features.t())  # [batch_size, batch_size]
        
        # 创建同类掩码
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        
        # 类内聚合损失（正样本对）
        pos_mask = label_matrix.fill_diagonal_(False)  # 排除自身
        pos_similarities = similarity_matrix[pos_mask]
        
        if len(pos_similarities) > 0:
            intra_loss = -torch.log(torch.exp(pos_similarities / self.temperature).sum() + 1e-8)
        else:
            intra_loss = torch.tensor(0.0, device=features.device)
        
        # 类间分离损失（负样本对）
        neg_mask = ~label_matrix
        neg_similarities = similarity_matrix[neg_mask]
        
        if len(neg_similarities) > 0:
            # 确保负样本对的距离足够大
            inter_loss = torch.mean(torch.clamp(self.margin - neg_similarities, min=0)**2)
        else:
            inter_loss = torch.tensor(0.0, device=features.device)
        
        total_loss = self.alpha * intra_loss + self.beta * inter_loss
        return total_loss
    

class IntraClassVarianceLoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        """
        类内方差最小化损失
        
        Args:
            temperature: 温度参数，控制分布的尖锐程度
            reduction: 损失 reduction 方式
        """
        super(IntraClassVarianceLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, features, labels):
        """
        Args:
            features: 特征张量 [batch_size, feat_dim]
            labels: 标签 [batch_size]
        """
        unique_labels = torch.unique(labels)
        variance_loss = 0.0
        valid_classes = 0
        
        for label in unique_labels:
            # 选择当前类别的所有特征
            class_mask = (labels == label)
            class_features = features[class_mask]
            
            if len(class_features) < 2:  # 至少需要2个样本计算方差
                continue
                
            # 计算类内特征的协方差矩阵
            centered_features = class_features - class_features.mean(dim=0)
            covariance_matrix = torch.mm(centered_features.t(), centered_features) / (len(class_features) - 1)
            
            # 计算方差（协方差矩阵的迹）
            variance = torch.trace(covariance_matrix)
            
            # 可选：使用特征值的和作为更稳定的方差度量
            # eigenvalues, _ = torch.lobpcg(covariance_matrix, k=min(10, self.feat_dim))
            # variance = eigenvalues.sum()
            
            variance_loss += variance
            valid_classes += 1
        
        if valid_classes == 0:
            return torch.tensor(0.0, device=features.device)
            
        variance_loss /= valid_classes
        
        return variance_loss