import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.fft import SiNet
from models.vit_base import Attention_LoRA_FFT
from utils.schedulers import CosineSchedule

# 辅助函数：生成 one-hot 标签
def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(1, targets.long().view(-1, 1), 1)
    return onehot

class DSCA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA_FFT):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1  
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

        if self.dataset == 'cifar100':
            self.logit_norm = None
        else:
            self.logit_norm = 0.1
        self.task_sizes = []

        # ===== Fly-CL Initialization =====
        self.expand_dim = self.args['expand_dim']
        self.embd_dim = self.args['embd_dim']
        self.synaptic_degree = self.args['synaptic_degree']
        
        # 初始化冻结的投影矩阵
        self.projection_matrix = torch.zeros(self.expand_dim, self.embd_dim)
        for row in range(self.expand_dim):
            selected_cols = torch.randperm(self.embd_dim)[:self.synaptic_degree]
            self.projection_matrix[row, selected_cols] = torch.randn(self.synaptic_degree)
        self.projection_matrix = self.projection_matrix.to(self._device).to_sparse()
        # self.projection_matrix_temp = self.projection_matrix.clone()
        
        # 初始化统计矩阵
        self._G = torch.zeros(self.expand_dim, self.expand_dim).to(self._device)
        # self._G_temp = torch.zeros(self.expand_dim, self.expand_dim).to(self._device)
        # _Q 需要随着类别增长而动态扩展，初始化为0大小
        self._Q = None 
        # self._Q_temp = None

        self._proj_class_means = None # 存储投影后的均值
        self._proj_class_covs = None  # 存储投影后的协方差 (建议存对角线以节省显存)

    def after_task(self):
    
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)

        if self.args["GCVD"] and hasattr(self, 'next_task_capacity'):
            logging.info(f"Applying dynamic capacity {self.next_task_capacity} for Task {self._cur_task}")
            for m in self._network.modules():
                if isinstance(m, Attention_LoRA_FFT):
                    # 调整当前任务的参数大小
                    m.resize_task_capacity(self._cur_task, self.next_task_capacity)

        # self._G = self._G_temp
        # self._Q = self._Q_temp
        # self.projection_matrix = self.projection_matrix_temp
        # 更新 _Q 矩阵大小以适应新类别
        if self._Q is None:
            self._Q = torch.zeros(self.expand_dim, self._total_classes).to(self._device)
        else:
            prev_classes = self._Q.shape[1]
            if prev_classes < self._total_classes:
                new_Q = torch.zeros(self.expand_dim, self._total_classes).to(self._device)
                new_Q[:, :prev_classes] = self._Q
                self._Q = new_Q

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        
        # [新增] 频谱侦察
        # 必须在 optimizer 初始化之前调用！因为我们会重置 Parameter
        # 这里的 train_loader 是新任务的数据
        # self.perform_spectral_scouting(self.train_loader)

        self._train(self.train_loader, self.test_loader)
        # ===== Update Fly-CL Statistics after backbone training =====
        # BiLoRA 对骨干网络进行了微调，因此需要在训练完成后提取特征更新 G 和 Q
        logging.info("Updating Fly-CL statistics (Streaming Ridge)...")
        self._update_flycl_statistics(self.train_loader)
        # ==========================================================
        if self.args["GCVD"]:
            self.adjust_capacity_for_next_task()

        self.clustering(self.train_loader)
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "coef_k" + "." + str(self._network.module.numtask - 1) in name and name.split('.')[-1]  == str(str(self._network.module.numtask - 1)):
                    param.requires_grad_(True)
                if "coef_v" + "." + str(self._network.module.numtask - 1) in name and name.split('.')[-1]  == str(str(self._network.module.numtask - 1)):
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "coef_k" + "." + str(self._network.numtask - 1) in name and name.split('.')[-1]  == str(str(self._network.numtask - 1)):
                    param.requires_grad_(True)
                if "coef_v" + "." + str(self._network.numtask - 1) in name and name.split('.')[-1]  == str(str(self._network.numtask - 1)):
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.args['SFT']:
            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    self._network(inputs, get_cur_feat=True)
                    # if i > 3: break
                # for module in self._network.modules():
                #     if isinstance(module, Attention_LoRA_FFT):
                #         module.save_task_statistics()
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA_FFT):
                        module.save_col_norms(self._cur_task)

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                labels = torch.index_select(targets, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                outputs = self._network(inputs)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, targets)

                features = outputs['features'].to(torch.float32)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        # import ipdb
        # ipdb.set_trace()
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes==self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))


        if check_diff:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
                # vectors, _ = self._extract_vectors_aug(idx_loader)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                    logging.info(log_info)
                    np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)

        if oracle:
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-5
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov            

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            class_cov = torch.tensor(np.cov(vectors.T), dtype=torch.float64)+torch.eye(class_mean.shape[-1])*1e-4
            # centered_vectors = vectors - class_mean
            # class_cov = torch.matmul(centered_vectors.T, centered_vectors) / (centered_vectors.size(0) - 1)+torch.eye(class_mean.shape[-1])*1e-4

            if check_diff:
                log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                logging.info(log_info)
                np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self._cur_task, class_idx), self._class_means[class_idx, :])
                # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

    # ================= [修改] 先收集特征做 GCV，再流式更新 =================
    def _update_flycl_statistics(self, loader):
        self._network.eval()
        if self.projection_matrix.device != self._device:
            self.projection_matrix = self.projection_matrix.to(self._device)

        # [新增] 确保 class_counts 在正确的设备上
        if self.class_counts.device != self._device:
            self.class_counts = self.class_counts.to(self._device)
            
        all_sparse_feats = []
        all_targets = []
        
        # 1. 收集所有数据特征（用于 GCV SVD）
        # 注意：如果显存爆了，可以只取前 N 个 batch
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # [新增] 更新类别样本计数
                for t in targets:
                    if t >= len(self.class_counts):
                        # 动态扩展 counts 数组
                        new_counts = torch.zeros(max(t.item() + 1, len(self.class_counts) * 2), device=self._device).long()
                        new_counts[:len(self.class_counts)] = self.class_counts
                        self.class_counts = new_counts
                    self.class_counts[t] += 1
                
                if isinstance(self._network, nn.DataParallel):
                    features = self._network.module.extract_vector(inputs)
                else:
                    features = self._network.extract_vector(inputs)

                # [关键修改] 强 L2 归一化
                # 确保每个样本在 Ridge 空间的贡献权重是相等的
                # features = F.normalize(features, p=2, dim=1)
                
                # Sparse Projection & Top-K
                expanded = torch.sparse.mm(self.projection_matrix, features.T).T
                k = int(self.expand_dim * self.args['coding_level'])
                values, indices = expanded.topk(k, dim=1, largest=True)
                sparse_feats = torch.zeros_like(expanded)
                sparse_feats.scatter_(1, indices, values)
                
                all_sparse_feats.append(sparse_feats)
                all_targets.append(targets)
                
        # 2. 拼接数据
        H = torch.cat(all_sparse_feats, dim=0) # [Total_Samples, Expand_Dim]
        Y_idx = torch.cat(all_targets, dim=0)
        Y = target2onehot(Y_idx, self._total_classes)
        
        # 3. 执行 GCV 选择最佳 Ridge Lambda
        logging.info(f"Running GCV on {H.shape[0]} samples...")
        # select_ridge_parameter 需要 Dense Tensor 进行 SVD
        # Fly-CL 的 features 维度通常很大 (10000)，但 SVD 在 N < M 时由 N 决定
        # 确保 H 是 Dense (Topk操作后本身就是 Dense Tensor，只是含很多0)
        self.best_ridge, self.current_gcv_score = self.select_ridge_parameter(H, Y)
        
        # 4. 更新全局统计量
        # G += H^T @ H
        # Q += H^T @ Y
        # 使用一次大矩阵乘法比循环 batch 更快
        self._G += H.T @ H
        self._Q += H.T @ Y

        self._G_temp = self._G.clone()
        self._Q_temp = self._Q.clone()
    
    def adjust_capacity_for_next_task(self):
        """
        [改进版] 基于 GCV 趋势比率动态调整容量 (Parameter-Free)。
        原理：比例控制器 (Proportional Control)。
        公式：New_Capacity = Old_Capacity * (Current_GCV / Average_GCV)
        """
        # 1. 维护 GCV 历史记录
        if not hasattr(self, 'gcv_history'):
            self.gcv_history = []
        
        if hasattr(self, 'current_gcv_score'):
            self.gcv_history.append(self.current_gcv_score)
            
        if len(self.gcv_history) == 0:
            return

        # 2. 计算统计量
        # 使用历史所有任务的平均值作为“期望难度”
        avg_gcv = np.mean(self.gcv_history)
        current_gcv = self.current_gcv_score
        
        # 获取当前实际使用的参数量
        current_n_frq = 0
        for m in self._network.modules():
            if isinstance(m, Attention_LoRA_FFT):
                current_n_frq = m.coef_k[self._cur_task].shape[0] # 获取当前任务实际大小
                break
        if current_n_frq == 0: current_n_frq = 3000 # Fallback
        
        # 3. [核心去参逻辑] 比例缩放
        # 如果 Current == Avg, ratio = 1.0 -> 容量不变
        # 如果 Current > Avg (难), ratio > 1.0 -> 容量自动增加
        # 如果 Current < Avg (易), ratio < 1.0 -> 容量自动减少
        
        ratio = current_gcv / (avg_gcv + 1e-8)
        ratio = pow(ratio, self.args["gamma"]) # 指数平滑 (指数为1.0即线性)
        
        # 为了防止震荡，可以加一个平滑系数（例如开根号），或者直接用线性
        # 这里直接用线性，因为它最符合直觉：难度翻倍，参数翻倍
        new_capacity = int(current_n_frq * ratio)
        
        # 4. 工程上的安全边界 (不是超参，是物理限制)
        # 上限：不能超过 expand_dim (或 dim*dim)，防止显存爆炸
        # 下限：不能太小，导致无法学习 (例如 256)
        max_cap = 10000 
        min_cap = 256
        new_capacity = max(min_cap, min(new_capacity, max_cap))
        
        action = "MAINTAIN"
        if new_capacity > current_n_frq: action = "INCREASE"
        elif new_capacity < current_n_frq: action = "DECREASE"
            
        logging.info(f"[GCV-Guided Parameter-Free] Current GCV: {current_gcv:.4f} (Avg: {avg_gcv:.4f}). "
                     f"Ratio: {ratio:.2f}. Action: {action}. Next Task Capacity: {new_capacity}")

        # 5. 暂存决定
        self.next_task_capacity = new_capacity