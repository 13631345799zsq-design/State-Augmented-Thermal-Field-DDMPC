import os, math
import pickle
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytorch_ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from _0_NODE_Model import PhysicsInformedCNN

RANDOM_SEED = 42

def set_random_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
# ============================================================ #
# 5. 增强版 Loss Map 绘制 (Professional Loss Visualization)
# ============================================================ #
def plot_loss_map(loss_train_history, loss_validation_history, save_dir, filename):
    loss_train_np = np.array(loss_train_history)
    loss_validation_np = np.array(loss_validation_history)
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(loss_train_np, label='Training Loss', color='#1f77b4', linewidth=2)
    ax.plot(loss_validation_np, label='Validation Loss', color='#d62728', linewidth=2)
    ax.set_yscale('log')
    ax.set_title('Training Loss History (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (Log Scale)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    # 保存
    save_path = os.path.join(save_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss Map 已保存至: {save_path}")
# ==========================================
# 3. 数据集 (全序列加载)
# ==========================================
class FullSequenceDataset(Dataset):
    def __init__(self, datapath, testsize, indices):
        # 1. 获取所有 pkl 文件
        self.files = [f for f in os.listdir(datapath) if f.endswith('.pkl')]
        # 2. 必须按照数字大小排序 (1, 2, 3...)
        self.files.sort(key=lambda x: int(x.split('.')[0]))
        self.files = [self.files[i] for i in indices]
        # ====================================================
        self.datapath = datapath
        self.testsize = testsize
        # 打印一下确认加载了哪些文件
        print(f"全序列数据集加载: {len(self.files)} 个文件 -> {self.files}")

    def __len__(self):
        return len(self.files)

    def map_progress(self, map_tensor):
        return F.interpolate(
            map_tensor.unsqueeze(0), size=self.testsize, mode='area'
        ).squeeze(0)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        with open(os.path.join(self.datapath, file_name), 'rb') as f:
            data = pickle.load(f)
        # 将所有场数据统一到训练分辨率
        data['time_list'] = torch.FloatTensor(data['time_list']).view(-1, 1).to(torch.float32)
        data['flux_matrix'] = torch.stack([self.map_progress(m) for m in data['flux_matrix']]).to(torch.float32)
        data['cooling_matrix'] = torch.stack([self.map_progress(m) for m in data['cooling_matrix']]).to(torch.float32)
        data['envtemp_list'] = torch.FloatTensor(data['envtemp_list']).view(-1, 1).to(torch.float32)
        data['temp_matrix'] = torch.stack([self.map_progress(m) for m in data['temp_matrix']]).to(torch.float32)
        return data
# ==========================================
# 4. 主训练逻辑
# ==========================================
# datapath = r'F:\0_Database\2_MCP_data\Data_transient'
datapath = r'/root/autodl-tmp/ATE_work/Data_transient'
savepath = os.path.dirname(__file__)

# Loss functions
criterion_SL1 = nn.SmoothL1Loss().cuda()
criterion_SSIM = pytorch_ssim.SSIM(window_size = 11).cuda()

# 数据加载 (Batch Size = 1, 因为每个文件时间步长可能不同)
testsize = (128, 128)
batch_size = 4

time_D_min, time_D_max = 0, 20
epochs = 1000
n_folds = 1

n = 1
data_volume = 120
training_data_volume = round(data_volume / math.sqrt(2) ** (n-1))
training_epochs = round(math.sqrt(2) ** (n-1) * epochs)

print(f"Training Data Volume: {training_data_volume}, Training Epochs: {training_epochs}")

training_start_time = time.time()
total_training_epochs = n_folds * training_epochs
for n_fold in range(5):
    fold_seed = RANDOM_SEED + n_fold
    set_random_seed(fold_seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(fold_seed)
    train_indices = list(range(0, training_data_volume))
    val_indices = list(range(120, 160))
    train_dataset = FullSequenceDataset(datapath, testsize, train_indices)
    val_dataset = FullSequenceDataset(datapath, testsize, val_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=dataloader_generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    foldpath = os.path.join(savepath, f'fold_result_{n_fold+1}')
    modelfile = os.path.join(foldpath, 'models_1')
    lossfile = os.path.join(foldpath, 'losses_1')
    os.makedirs(modelfile, exist_ok=True)
    os.makedirs(lossfile, exist_ok=True)

    # 预训练模型路径 (请确保这里是你已经训练好的路径)
    model = PhysicsInformedCNN().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,min_lr=1e-5)

    print("开始递归微调...")

    train_losses_G = []
    val_losses_G = []
    for epoch in range(training_epochs):
        # train_part======================================================================== # 
        model.train() 
        train_loss_G = torch.zeros((), device='cuda')
        # 遍历所有文件 (每个文件是一个 Batch, size=1)
        for file_idx, data in enumerate(train_dataloader):
            time_seq  = data['time_list'].cuda()         # [B, T, 1]
            flux_seq  = data['flux_matrix'].cuda()         # [B, T, 64, 64]
            cool_seq  = data['cooling_matrix'].cuda()      # [B, T, 64, 64]
            envtemp_seq = data['envtemp_list'].cuda()   # [B, T, 64, 64]
            gt_maps   = data['temp_matrix'].cuda()  # [B, T, 64, 64]
            B = time_seq.shape[0]
            # 初始状态 [B, 1, 64, 64]
            m_curr = gt_maps[:, 0, :, :]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0
            file_loss = torch.zeros((), device='cuda')
            steps = time_seq.shape[1] - 1
            # === 递归时间步 (并行跑 B 个文件) ===
            for t in range(steps):
                # 1. 准备并行输入 [B, 1, 64, 64]
                dt = time_seq[:, t+1] - time_seq[:, t] # [B, 1]
                dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                time_D_m = dt_norm.view(B, 1, 1, 1).expand(B, 1, testsize[0], testsize[1])
                flux_m = flux_seq[:, t]     # [B, 1, 64, 64]
                cooling_m = cool_seq[:, t]
                envtemp_m = envtemp_seq[:, t].view(B, 1, 1, 1).expand(B, 1, testsize[0], testsize[1])
                m_gt_next = gt_maps[:, t+1]
                # 2. 前向传播
                with torch.amp.autocast("cuda"):
                    m_next_pred = model(time_D_m, m_curr, flux_m, cooling_m, envtemp_m)
                m_next_pred = m_next_pred.float()
                # 3. 计算 Loss
                loss_step = 10 * criterion_SL1(m_next_pred, m_gt_next) + \
                            1 * (1 - criterion_SSIM(m_next_pred, m_gt_next))
                batch_loss += loss_step
                file_loss += loss_step.detach()
                # TBPTT 梯度截断逻辑
                is_window_end = ((t + 1) % 16 == 0) or (t == steps - 1)
                if is_window_end:
                    scaler.scale(batch_loss).backward()
                    if t == steps - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    m_curr = m_next_pred.detach()
                    batch_loss = 0
                else:
                    m_curr = m_next_pred

            # 计算该文件的平均单步 Loss
            train_loss_G += file_loss / steps
            file_loss_value = (file_loss / steps).item()
            print(f"Epoch {epoch+1}, Batch {batch_size*(file_idx+1)}/{batch_size*len(train_dataloader)}, Loss: {file_loss_value:.6f}")
        train_losses_G.append((train_loss_G / len(train_dataloader)).item())
        scheduler.step(train_losses_G[-1])
        print(f"==> Epoch {epoch+1} Finished.")
        print(f"    Total: {train_losses_G[-1]:.6f}")

        # validation_part======================================================================== # 
        with torch.no_grad():
            model.eval()  # 切换到评估模式
            val_loss_G = torch.zeros((), device='cuda')
            # 遍历所有文件 (每个文件是一个 Batch, size=1)
            for file_idx, data in enumerate(val_dataloader):
                time_seq  = data['time_list'].cuda()         # [B, T, 1]
                flux_seq  = data['flux_matrix'].cuda()         # [B, T, 64, 64]
                cool_seq  = data['cooling_matrix'].cuda()      # [B, T, 64, 64]
                envtemp_seq = data['envtemp_list'].cuda()   # [B, T, 64, 64]
                gt_maps   = data['temp_matrix'].cuda()  # [B, T, 64, 64]
                B = time_seq.shape[0]
                # 初始状态 [B, 1, 64, 64]
                m_curr = gt_maps[:, 0, :, :]
                file_loss = torch.zeros((), device='cuda')
                steps = time_seq.shape[1] - 1
                for t in range(steps):
                    # 1. 准备并行输入 [B, 1, 64, 64]
                    dt = time_seq[:, t+1] - time_seq[:, t] # [B, 1]
                    dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                    time_D_m = dt_norm.view(B, 1, 1, 1).expand(B, 1, testsize[0], testsize[1])
                    flux_m = flux_seq[:, t]     # [B, 1, 64, 64]
                    cooling_m = cool_seq[:, t]
                    envtemp_m = envtemp_seq[:, t].view(B, 1, 1, 1).expand(B, 1, testsize[0], testsize[1])
                    m_gt_next = gt_maps[:, t+1]
                    # 2. 前向传播
                    with torch.amp.autocast("cuda"):
                        m_next_pred = model(time_D_m, m_curr, flux_m, cooling_m, envtemp_m)
                    m_next_pred = m_next_pred.float()
                    m_curr = m_next_pred
                    # 3. 计算 Loss
                    loss_step = 10 * criterion_SL1(m_next_pred, m_gt_next) + \
                                1 * (1 - criterion_SSIM(m_next_pred, m_gt_next))
                    # 累积统计数据
                    file_loss += loss_step.detach()
                # 计算该文件的平均单步 Loss
                val_loss_G += file_loss / steps
            val_losses_G.append((val_loss_G / len(val_dataloader)).item())
            
            # 保存模型
            if epoch == 0:
                best_score = val_losses_G[-1]
            else:
                if val_losses_G[-1] < best_score:
                    best_score = val_losses_G[-1]
                    if epoch >= 500:  # 至少训练 300 个 epoch 后再保存模型
                        torch.save(model.state_dict(), os.path.join(modelfile, f'model_epoch_{epoch+1}.pth'))

        # loss存储为numpy文件
        np.save(os.path.join(lossfile, 'Glosses_train_%d.npy' % (epoch+1)), train_losses_G)
        np.save(os.path.join(lossfile, 'Glosses_val_%d.npy' % (epoch+1)), val_losses_G)

        completed_epochs = n_fold * epochs + epoch + 1
        elapsed_seconds = time.time() - training_start_time
        remaining_seconds = elapsed_seconds / completed_epochs * (total_training_epochs - completed_epochs)
        estimated_total_seconds = elapsed_seconds + remaining_seconds
        print(f"Training progress [{completed_epochs}/{total_training_epochs}] - "
              f"Elapsed: {timedelta(seconds=int(elapsed_seconds))}, "
              f"Remaining: {timedelta(seconds=int(remaining_seconds))}, "
              f"Estimated total: {timedelta(seconds=int(estimated_total_seconds))}")

    # 保存loss曲线图  
    plot_loss_map(train_losses_G, val_losses_G, foldpath, f'transient_loss_{epoch+1}.png')

os.system("/usr/bin/shutdown")
