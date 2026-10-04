import math
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_ssim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from _0_NODE_GRU_Model import PhysicsInformedCNN


class FullSequenceDataset(Dataset):
    def __init__(self, datapath, testsize, indices):
        files = [name for name in os.listdir(datapath) if name.endswith('.pkl')]
        files.sort(key=lambda name: int(os.path.splitext(name)[0]))
        self.files = [files[index] for index in indices]
        self.datapath = datapath
        self.testsize = testsize
        print(f"全序列数据集加载: {len(self.files)} 个文件 -> {self.files}")

    def __len__(self):
        return len(self.files)

    def map_progress(self, map_tensor):
        return F.interpolate(
            map_tensor.unsqueeze(0), size=self.testsize, mode='area'
        ).squeeze(0)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        with open(os.path.join(self.datapath, file_name), 'rb') as file:
            data = pickle.load(file)

        data['time_list'] = torch.as_tensor(
            data['time_list'], dtype=torch.float32
        ).view(-1, 1)
        data['flux_matrix'] = torch.stack(
            [self.map_progress(item) for item in data['flux_matrix']]
        ).to(torch.float32)
        data['cooling_matrix'] = torch.stack(
            [self.map_progress(item) for item in data['cooling_matrix']]
        ).to(torch.float32)
        data['envtemp_list'] = torch.as_tensor(
            data['envtemp_list'], dtype=torch.float32
        ).view(-1, 1)
        data['temp_matrix'] = torch.stack(
            [self.map_progress(item) for item in data['temp_matrix']]
        ).to(torch.float32)
        return data


def chip_temp_detect(temp_matrix, chip_loc, testsize):
    temp_matrix = temp_matrix.squeeze()
    chiptemp_list = []
    for x_loc, z_loc, width, depth in chip_loc:
        chip_loc_x = math.ceil(testsize[0] * (x_loc - 1) / 180)
        chip_loc_z = math.ceil(testsize[1] * (z_loc - 1) / 135)
        chip_loc_w = math.ceil(testsize[0] * width / 180)
        chip_loc_d = math.ceil(testsize[1] * depth / 135)
        x = math.ceil(chip_loc_x + chip_loc_w / 2)
        z = math.ceil(chip_loc_z + chip_loc_d / 2)
        chiptemp_list.append(temp_matrix[z, x].item())
    return chiptemp_list


def find_latest_model_epoch(model_dir):
    epochs = []
    for file_name in os.listdir(model_dir):
        match = re.fullmatch(r'model_epoch_(\d+)\.pth', file_name)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f'未在 {model_dir} 中找到 model_epoch_*.pth')
    return max(epochs)


datapath = r'F:\0_Database\2_MCP_data\Data_transient'
savepath = os.path.dirname(__file__)
testsize = (128, 128)
test_indices = list(range(180, 200))
test_dataset = FullSequenceDataset(datapath, testsize, test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

criterion_SSIM = pytorch_ssim.SSIM(window_size=11).cuda()

chip_loc = [
    [65, 55, 50, 50], [15, 90, 30, 30], [142, 13.5, 20, 20],
    [142, 43.5, 20, 20], [142, 73.5, 20, 20], [142, 103.5, 20, 20],
    [115, 20, 10, 10], [90, 20, 10, 10], [65, 20, 10, 10],
    [40, 20, 10, 10], [22, 40, 10, 10], [22, 65, 10, 10],
]
point_labels = [f'chip{i}' for i in range(1, 13)]
time_D_min, time_D_max = 0, 20
n_folds = 5
model_epoch = None  # None 表示自动加载最新模型

Fold_list = []
RMSE_list = []
SSIM_list = []
MaxAEH_list = []
MaxAEF_list = []

with torch.no_grad():
    for n_fold in range(n_folds):
        fold_index = f'fold_{n_fold + 1}'
        foldpath = os.path.join(savepath, f'fold_result_{n_fold + 1}')
        modelfile = os.path.join(foldpath, 'models_1')
        resultfile = os.path.join(foldpath, 'results_1')
        os.makedirs(resultfile, exist_ok=True)

        epoch_to_load = (
            find_latest_model_epoch(modelfile)
            if model_epoch is None else model_epoch
        )
        model_path = os.path.join(modelfile, f'model_epoch_{epoch_to_load}.pth')
        model = PhysicsInformedCNN().cuda()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print(f'\n=== Testing {fold_index}: model_epoch_{epoch_to_load}.pth ===')

        curve_save_dir = os.path.join(resultfile, 'temperature_curves')
        os.makedirs(curve_save_dir, exist_ok=True)

        fold_rmse_sum = 0.0
        fold_ssim_sum = 0.0
        fold_maxaeh_sum = 0.0
        fold_maxaef_sum = 0.0
        rmse_by_time = None
        ssim_by_time = None
        maxaeh_by_time = None
        maxaef_by_time = None

        for file_idx, data in enumerate(test_dataloader):
            case_name = os.path.splitext(test_dataset.files[file_idx])[0]
            file_res_dir = os.path.join(resultfile, case_name)
            os.makedirs(file_res_dir, exist_ok=True)

            time_seq = data['time_list'].cuda()
            flux_seq = data['flux_matrix'].cuda()
            cool_seq = data['cooling_matrix'].cuda()
            envtemp_seq = data['envtemp_list'].cuda()
            gt_maps = data['temp_matrix'].cuda()
            batch_size = time_seq.shape[0]
            steps = time_seq.shape[1] - 1

            m_curr = gt_maps[:, 0]
            h_old = None
            pred_curves = []
            gt_curves = []
            file_rmse = []
            file_ssim = []
            file_maxaeh = []
            file_maxaef = []

            for t in range(steps):
                dt = time_seq[:, t + 1] - time_seq[:, t]
                dt_norm = (dt - time_D_min) / (time_D_max - time_D_min)
                time_D_m = dt_norm.view(batch_size, 1, 1, 1).expand(
                    batch_size, 1, *testsize
                )
                flux_m = flux_seq[:, t]
                cooling_m = cool_seq[:, t]
                envtemp_m = envtemp_seq[:, t].view(
                    batch_size, 1, 1, 1
                ).expand(batch_size, 1, *testsize)
                m_next_gt = gt_maps[:, t + 1]

                m_next_pred, h_curr = model(
                    time_D_m, m_curr, flux_m, cooling_m, envtemp_m, h_old
                )
                m_curr = m_next_pred
                h_old = h_curr

                rmse_step = torch.sqrt(F.mse_loss(m_next_pred, m_next_gt)).item()
                ssim_step = (1 - criterion_SSIM(m_next_pred, m_next_gt)).item()
                file_rmse.append(rmse_step)
                file_ssim.append(ssim_step)

                pred_points = chip_temp_detect(m_next_pred, chip_loc, testsize)
                gt_points = chip_temp_detect(m_next_gt, chip_loc, testsize)
                maxaeh = np.max(np.abs(np.array(pred_points) - np.array(gt_points)))
                maxaef = torch.max(torch.abs(m_next_pred - m_next_gt)).item()
                file_maxaeh.append((85 - 20) * maxaeh)
                file_maxaef.append((85 - 20) * maxaef)
                pred_curves.append(pred_points)
                gt_curves.append(gt_points)

                if t % 5 == 0 or t == steps - 1:
                    pred_map = m_next_pred.squeeze().cpu().numpy()
                    gt_map = m_next_gt.squeeze().cpu().numpy()
                    diff = pred_map - gt_map
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    images = [
                        axes[0].imshow(pred_map, cmap='inferno', vmin=0, vmax=1),
                        axes[1].imshow(gt_map, cmap='inferno', vmin=0, vmax=1),
                        axes[2].imshow(diff, cmap='bwr', vmin=-0.1, vmax=0.1),
                    ]
                    axes[0].set_title(f'Pred (t={t + 1})')
                    axes[1].set_title(f'GT (t={t + 1})')
                    axes[2].set_title(f'Diff (MAE={np.mean(np.abs(diff)):.4f})')
                    for axis, image in zip(axes, images):
                        fig.colorbar(image, ax=axis)
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(file_res_dir, f'step_{t + 1:04d}.png'), dpi=80
                    )
                    plt.close(fig)

            file_rmse = np.asarray(file_rmse)
            file_ssim = np.asarray(file_ssim)
            file_maxaeh = np.asarray(file_maxaeh)
            file_maxaef = np.asarray(file_maxaef)
            fold_rmse_sum += file_rmse.mean()
            fold_ssim_sum += file_ssim.mean()
            fold_maxaeh_sum += file_maxaeh.mean()
            fold_maxaef_sum += file_maxaef.mean()

            if rmse_by_time is None:
                rmse_by_time = file_rmse
                ssim_by_time = file_ssim
                maxaeh_by_time = file_maxaeh
                maxaef_by_time = file_maxaef
            else:
                rmse_by_time += file_rmse
                ssim_by_time += file_ssim
                maxaeh_by_time += file_maxaeh
                maxaef_by_time += file_maxaef

            pred_curves = np.asarray(pred_curves)
            gt_curves = np.asarray(gt_curves)
            x_axis_data = time_seq[0, 1:, 0].cpu().numpy()
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            for i, axis in enumerate(axes.flatten()):
                axis.plot(x_axis_data, gt_curves[:, i], 'k-', label='GT', alpha=0.6)
                axis.plot(x_axis_data, pred_curves[:, i], 'r--', label='Pred')
                residual = np.abs(gt_curves[:, i] - pred_curves[:, i])
                residual_axis = axis.twinx()
                residual_axis.plot(x_axis_data, residual, 'b:', label='Residual')
                residual_axis.set_ylim(0, 0.02)
                point_rmse = np.sqrt(np.mean(residual ** 2))
                axis.set_title(f'{point_labels[i]}\nRMSE: {point_rmse:.4f}')
                axis.set_ylim(0, 1)
                axis.grid(True, alpha=0.3)
                if i == 0:
                    lines, labels = axis.get_legend_handles_labels()
                    residual_lines, residual_labels = residual_axis.get_legend_handles_labels()
                    axis.legend(lines + residual_lines, labels + residual_labels)
            fig.suptitle(f'Temperature Curves - {case_name}')
            fig.tight_layout()
            fig.savefig(
                os.path.join(curve_save_dir, f'temperature_curves_{case_name}.png'),
                dpi=150,
            )
            plt.close(fig)

        case_count = len(test_dataloader)
        fold_rmse = fold_rmse_sum / case_count
        fold_ssim = fold_ssim_sum / case_count
        fold_maxaeh = fold_maxaeh_sum / case_count
        fold_maxaef = fold_maxaef_sum / case_count
        print(f'RMSELoss: {fold_rmse:.5f}')
        print(f'SSIMLoss: {fold_ssim:.5f}')
        print(f'MaxAEH: {fold_maxaeh:.5f} °C')
        print(f'MaxAEF: {fold_maxaef:.5f} °C')
        Fold_list.append(fold_index)
        RMSE_list.append(fold_rmse)
        SSIM_list.append(fold_ssim)
        MaxAEH_list.append(fold_maxaeh)
        MaxAEF_list.append(fold_maxaef)

        time_values = time_seq[0, 1:, 0].cpu().numpy()
        pd.DataFrame({
            'TIME': time_values,
            'RMSE': rmse_by_time / case_count,
            'SSIM': ssim_by_time / case_count,
            'MaxAEH': maxaeh_by_time / case_count,
            'MaxAEF': maxaef_by_time / case_count,
        }).to_csv(
            os.path.join(savepath, f'All_results_with_time_fold_{n_fold + 1}.csv'),
            index=False,
        )

pd.DataFrame({
    'Fold': Fold_list,
    'RMSE': RMSE_list,
    'SSIM': SSIM_list,
    'MaxAEH': MaxAEH_list,
    'MaxAEF': MaxAEF_list,
}).to_csv(os.path.join(savepath, 'All_results.csv'), index=False)
