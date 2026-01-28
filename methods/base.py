import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import optim
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args['memory_size']
        self._memory_per_class = args['memory_per_class']
        self._fixed_memory = args['fixed_memory']
        self._device = args['device'][0]
        self._multiple_gpus = args['device']

        # [新增] 存储最佳 Ridge 参数，默认值设为 1e4，后续会被 GCV 覆盖
        self.best_ridge = 1e4 
        # [新增] 记录每个类别的样本数量，用于 SDC 更新 Ridge Q 矩阵
        self.class_counts = torch.zeros(args.get('num_classes', 1000)).long() # 预设一个足够大的长度或者动态扩展

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, 'Total classes is 0'
            return (self._memory_size // self._total_classes)

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': self._network.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
                                                   decimals=2)

        return ret

    def eval_task(self):
        y_pred, y_pred_with_task, y_true, y_pred_task, y_true_task = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        cnn_accy_with_task = self._evaluate(y_pred_with_task, y_true)
        cnn_accy_task = (y_pred_task == y_true_task).sum().item()/len(y_pred_task)

        if hasattr(self, '_class_means'):
            _class_means = self._class_means
            y_pred, y_true = self._eval_nme(self.test_loader, _class_means)
            nme_accy = self._evaluate(y_pred[:,0], y_true)
        else:
            nme_accy = None

        ridge_accy = None
        if hasattr(self, '_G') and hasattr(self, '_Q'):
            ridge_weights = self._compute_ridge_weights()
            # ridge_weights = self._compute_ridge_weights_expert()
            y_pred_ridge, y_true_ridge = self._eval_ridge(self.test_loader, ridge_weights)
            ridge_accy = self._evaluate(y_pred_ridge[:,0], y_true_ridge)

        return cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task, ridge_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        class_means = F.normalize(torch.tensor(class_means), p=2, dim=-1)
        dists = cdist(class_means.cpu().numpy(), vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _compute_ridge_weights(self):
        """计算岭回归权重 w = (G + lambda*I)^-1 Q"""
        G = self._G.to(self._device)
        Q = self._Q.to(self._device)
        
        # 使用通过 GCV 选出的最佳参数
        ridge_lambda = self.best_ridge
        
        regularizer = ridge_lambda * torch.eye(G.size(0)).to(self._device)
        
        # 使用 Cholesky 分解加速求解正定方程组
        try:
            L = torch.linalg.cholesky(G + regularizer)
            W_out = torch.cholesky_solve(Q, L)
        except RuntimeError:
            # 如果数值不稳定，回退到求逆
            print("Warning: Cholesky failed, using lstsq or inv")
            W_out = torch.linalg.solve(G + regularizer, Q)
            
        return W_out

    def _compute_ridge_weights_expert(self):
        """
        专家级方案：基于 SVD 截断的岭回归权重计算
        解决 N approx D 时特征值过小导致的数值爆炸
        [修改] 使用 svd 替代 eigh，并不再使用缓存
        """
        G = self._G.to(self._device)
        Q = self._Q.to(self._device)
        
        # 1. 对 G 进行 SVD 分解
        # G = U * S * Vh
        # G 是协方差矩阵，理论上 U 和 Vh.T 是相同的（特征向量），S 是奇异值（特征值）
        try:
            # 使用 svd
            U, S, Vh = torch.linalg.svd(G)
        except RuntimeError:
            logging.warning("SVD failed, falling back to Cholesky solve.")
            return self._compute_ridge_weights() # 回退到旧方法

        # 2. 谱截断 (Spectral Clipping)
        # 设定一个阈值，小于该阈值的特征值被认为是“危险的噪声”
        # 经验阈值：最大特征值的 1e-6 倍
        threshold = S.max() * 1e-6
        
        # 3. 计算逆奇异值
        # 使用你 GCV 选出来的最佳 lambda
        ridge_lambda = self.best_ridge 
        
        # 构造对角逆矩阵向量
        inv_s = torch.zeros_like(S)
        mask = S > threshold
        
        # 关键点：只对“健康”的特征值求逆，彻底切断噪声方向
        # 公式: 1 / (sigma + lambda)
        inv_s[mask] = 1.0 / (S[mask] + ridge_lambda)
        
        # 4. 重构权重 W = (G + lambda*I)^-1 @ Q
        # 利用 SVD 性质: G^-1 = Vh.T @ diag(1/S) @ U.T
        # 所以 W = Vh.T @ diag(inv_s) @ U.T @ Q
        
        # 步骤 A: Temp = U.T @ Q  --> (D, C)
        temp = U.T @ Q
        
        # 步骤 B: Scale by inverse eigenvalues (Broadcasting) --> (D, C)
        temp = inv_s.unsqueeze(1) * temp
        
        # 步骤 C: W = Vh.T @ Temp --> (D, C)
        W_out = Vh.T @ temp
        
        # [关键补充] 方案三：权重模长对齐 (Weight Alignment)
        # 解决新旧任务权重 Scale 不一致导致 Acc 曲线中间凹陷的问题
        if self._known_classes > 0 and False:
            old_W = W_out[:, :self._known_classes]
            new_W = W_out[:, self._known_classes:]
            
            # 只有当新旧类别都存在时才对齐
            if new_W.shape[1] > 0 and old_W.shape[1] > 0:
                old_norm = torch.norm(old_W, p=2, dim=0).mean()
                new_norm = torch.norm(new_W, p=2, dim=0).mean()
                
                if new_norm > 1e-8:
                    gamma = old_norm / new_norm
                    # 仅做温和的对齐，防止矫枉过正
                    W_out[:, self._known_classes:] *= gamma
                    # logging.info(f"Weight Alignment Applied: gamma={gamma:.4f}")

        return W_out
    
    # --- [新增] Fly-CL GCV 核心算法 ---
    def select_ridge_parameter(self, Features, Y, ridge_lower=6, ridge_upper=10):
        """
        利用 GCV 选择最佳 lambda，并返回对应的 GCV score
        """
        X = Features
        n_samples = X.shape[0]
        
        # SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S_sq = S**2
        UTY = U.T @ Y
        
        ridges = torch.tensor(10.0 ** np.arange(ridge_lower, ridge_upper), device=X.device)
        # ridges = torch.tensor(10.0 * np.arange(ridge_lower, ridge_upper), device=X.device)
        
        gcv_scores = []
        for ridge in ridges:
            # GCV 公式
            diag = S_sq / (S_sq + ridge)
            df = diag.sum()
            Y_hat = U @ (diag[:, None] * UTY)
            residual = torch.norm(Y - Y_hat)**2
            gcv = (residual / n_samples) / (1 - df / n_samples)**2
            gcv_scores.append(gcv.item())

        # 选择最佳
        optimal_idx = np.argmin(gcv_scores)
        best_lambda = ridges[optimal_idx].item()
        best_gcv = gcv_scores[optimal_idx] # 获取最佳 GCV 分数
        
        logging.info(f"GCV Selected Ridge Lambda: {best_lambda:.2e}, GCV Score: {best_gcv:.4f}")
        
        # 返回两个值
        return best_lambda, best_gcv

    def _eval_ridge(self, loader, ridge_weights):
        """Fly-CL 风格的评估: 提取特征 -> 稀疏投影 -> TopK -> 岭回归预测"""
        self._network.eval()
        
        # 1. 提取特征 (N, D)
        vectors, y_true = self._extract_vectors(loader)
        vectors_t = torch.tensor(vectors, dtype=torch.float32).to(self._device)
        # [关键修改] 测试时也要归一化
        # vectors_t = F.normalize(vectors_t, p=2, dim=1) 
        
        # 2. 稀疏随机投影 & Top-k
        if not hasattr(self, 'projection_matrix'):
            raise RuntimeError("projection_matrix not initialized for Fly-CL Ridge")
            
        # 投影: (Expand_Dim, Emb_Dim) @ (Emb_Dim, N) -> (Expand_Dim, N)
        # 注意 transpose 处理
        expanded = torch.sparse.mm(self.projection_matrix, vectors_t.T).T # (N, Expand_Dim)
        
        # Top-k 稀疏化
        k = int(self.args['expand_dim']* self.args['coding_level'])
        values, indices = expanded.topk(k, dim=1, largest=True)
        sparse_feats = torch.zeros_like(expanded)
        sparse_feats.scatter_(1, indices, values)
        
        # 3. 预测 (N, Expand_Dim) @ (Expand_Dim, Num_Classes) -> (N, Num_Classes)
        logits = torch.mm(sparse_feats, ridge_weights)
        
        # 4. 获取 Top-k 预测 (排序为降序，得分越高越好)
        scores = logits.cpu().numpy()
        # argsort 默认升序，取反后取前 topk
        y_pred = np.argsort(scores, axis=1)[:, ::-1][:, :self.topk]
        
        return y_pred, y_true
    # ==============================

    def _extract_vectors(self, loader, task=None):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device), task=task))
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device), task=task))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    def _get_exemplar_with_class_idxes(self, class_idx):
        ex_d, ex_t = np.array([]), np.array([])
        # class_idx = [i for i in class_idx]
        for i in class_idx:
            mask = np.where(self._targets_memory == i)[0]
            ex_d = np.concatenate((ex_d, copy.deepcopy(self._data_memory[mask]))) if len(ex_d) != 0 \
                else copy.deepcopy(self._data_memory[mask])
            ex_t = np.concatenate((ex_t, copy.deepcopy(self._targets_memory[mask]))) if len(ex_t) != 0 \
                else copy.deepcopy(self._targets_memory[mask])
        return ex_d, ex_t