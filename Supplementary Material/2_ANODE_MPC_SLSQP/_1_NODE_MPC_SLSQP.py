import os, time, math, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import minimize
from _0_NODE_Model import PhysicsInformedCNN

# --- 参数设定 ---------------------------------------------------- #
model_epoch = 980
testsize = (128, 128)
chip_power = [30, 10, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3]
target_temp = 65.0
env_temp = 40.0
initial_temp = 40.0
temp_min, temp_max = 20.0, 85.0
flux_min, flux_max = 0.0, 0.03

# 最小风扇转速下首步最大升温速率的比例
T_Grad_ratio = 0.01
time_total = 1000
time_D_max = 20.0

c_max, c_min = 6300.0, 0.0
fan_max, fan_min = 6300.0, 1300.0
fan_start = 3000.0

time_step_l = [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]
time_step = 0.5

hidden_dim = 4
optimizer_iterations = 20
optimizer_ftol = 1e-5
heat_epsilon = 0.05 / 80.0
grad_epsilon = heat_epsilon / time_step
weight_heat = 1e7
weight_grad = 5e2
weight_cool = 5e-3
save_interval = 10

# 模型加载 ---------------------------------------------------- #
path = os.path.dirname(__file__)
transient_model_dir = os.path.join(path, 'models_1')
savepath = os.path.join(
    path,
    f'SLSQP_results_C{weight_cool}_H{weight_heat}_G{weight_grad:g}_'
    f'{initial_temp:g}_{target_temp:g}',
)
os.makedirs(savepath, exist_ok=True)

transient_model = PhysicsInformedCNN().cuda()
transient_model.load_state_dict(torch.load(os.path.join(transient_model_dir, f'model_epoch_{model_epoch}.pth'), weights_only=True))
transient_model.eval()
for parameter in transient_model.parameters():
    parameter.requires_grad_(False)
# ============================================================ #
# 1. 训练主流程
# ============================================================ #
def chip_temp_detect(temp_matrix, chip_loc, testsize):
    temp_matrix = temp_matrix.squeeze() # [H, W]
    chiptemp_list = []
    for i in range(12):
        chip_loc_x = math.ceil(testsize[0]*(chip_loc[i][0]-1)/180)
        chip_loc_z = math.ceil(testsize[1]*(chip_loc[i][1]-1)/135)
        chip_loc_w = math.ceil(testsize[0]*chip_loc[i][2]/180)
        chip_loc_d = math.ceil(testsize[1]*chip_loc[i][3]/135)
        # 取预测温度值
        x = math.ceil(chip_loc_x + chip_loc_w/2)
        z = math.ceil(chip_loc_z + chip_loc_d/2)
        chiptemp_list.append(temp_matrix[z,x].item())
    return chiptemp_list

chip_loc = [[65, 55, 50, 50], [15, 90, 30, 30], [142, 13.5, 20, 20], [142, 43.5, 20, 20],
            [142, 73.5, 20, 20], [142, 103.5, 20, 20], [115, 20, 10, 10], [90, 20, 10, 10], 
            [65, 20, 10, 10], [40, 20, 10, 10], [22, 40, 10, 10], [22, 65, 10, 10]]


def prepare_flux_matrix(chip_loc, chip_power, testsize):
    flux_matrix = torch.zeros((1, 1, 1350, 1800), dtype=torch.float32)
    for (x_loc, z_loc, width, depth), power in zip(chip_loc, chip_power):
        x1 = math.ceil(x_loc * 10)
        z1 = math.ceil(z_loc * 10)
        x2 = math.ceil(x1 + width * 10)
        z2 = math.ceil(z1 + depth * 10)
        flux = (power / (width * depth) - flux_min) / (flux_max - flux_min)
        flux_matrix[:, :, z1:z2, x1:x2] = flux
    return F.interpolate(
        flux_matrix[:, :, 10:-10, 10:-10], size=testsize, mode='area'
    )


_chip_z_idx = []
_chip_x_idx = []
for i in range(12):
    chip_loc_x = math.ceil(testsize[0]*(chip_loc[i][0]-1)/180)
    chip_loc_z = math.ceil(testsize[1]*(chip_loc[i][1]-1)/135)
    chip_loc_w = math.ceil(testsize[0]*chip_loc[i][2]/180)
    chip_loc_d = math.ceil(testsize[1]*chip_loc[i][3]/135)
    x = math.ceil(chip_loc_x + chip_loc_w/2)
    z = math.ceil(chip_loc_z + chip_loc_d/2)
    _chip_x_idx.append(x)
    _chip_z_idx.append(z)
CHIP_Z_IDX = torch.tensor(_chip_z_idx, dtype=torch.long, device='cuda')
CHIP_X_IDX = torch.tensor(_chip_x_idx, dtype=torch.long, device='cuda')
# 在 MPC_Optimizer 的 __init__ 中添加
# ============================================================ #
# 1. 预先构建全局 Mask (只执行一次)
# ============================================================ #
def prepare_masks(imgSize):
    # 物理坐标
    cooling_loc = [[50-25, 63.5-25, 50, 50], [130-25, 63.5-25, 50, 50]]
    # 创建两个独立的掩码，形状为 [1, 1, H, W]
    mask1 = torch.zeros((1, 1, imgSize[0]*10, imgSize[1]*10), device='cuda', dtype=torch.float32)
    mask2 = torch.zeros((1, 1, imgSize[0]*10, imgSize[1]*10), device='cuda', dtype=torch.float32)
    # 风扇 1 区域
    x1, z1, w, d = cooling_loc[0]
    mask1[:, :, math.ceil(z1*10):math.ceil((z1+d)*10), math.ceil(x1*10):math.ceil((x1+w)*10)] = 1.0
    # 风扇 2 区域
    x1, z1, w, d = cooling_loc[1]
    mask2[:, :, math.ceil(z1*10):math.ceil((z1+d)*10), math.ceil(x1*10):math.ceil((x1+w)*10)] = 1.0
    # 3. 物理边缘裁剪
    crop_p = 10 
    # 4. 下采样至目标尺寸 (256, 256)
    mask1 = F.interpolate(mask1[:, :, crop_p:-crop_p, crop_p:-crop_p], 
                              size=testsize, mode='area')
    mask2 = F.interpolate(mask2[:, :, crop_p:-crop_p, crop_p:-crop_p], 
                              size=testsize, mode='area')
    return mask1, mask2

# 全局调用
global_mask1, global_mask2 = prepare_masks((135, 180))
global_mask1 = global_mask1.cuda()
global_mask2 = global_mask2.cuda()
# ============================================================ #
# 2. 绝对安全的函数式冷却矩阵生成 
# ============================================================ #
def differentiable_cooling_matrix(u_plan, mask1, mask2, testsize=(128, 128)):
    b, n, _ = u_plan.shape
    # 直接基于 u_plan 操作，它本身就在 GPU 上
    u1 = u_plan[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    u2 = u_plan[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    # 张量乘法，100% 保持梯度
    matrix = (u1 * mask1) + (u2 * mask2)
    final_matrix = matrix.view(b, n, 1, testsize[0], testsize[1])
    return final_matrix
# ============================================================ #
# 3. 优化器类定义
# ============================================================ #
class MPC_Optimizer:
    def __init__(self, flux_m, envtemp_m, target_temperature, T_Grad_target, time_step_l, iterations):
        self.u_max = torch.tensor([(fan_max - c_min) / (c_max - c_min)], device='cuda', dtype=torch.float32)
        self.u_min = torch.tensor([(fan_min - c_min) / (c_max - c_min)], device='cuda', dtype=torch.float32)
        self.flux_m = flux_m.detach()
        self.envtemp_m = envtemp_m.detach()
        self.time_step_l = torch.tensor(time_step_l, device='cuda', dtype=torch.float32).view(1, -1, 1)
        self.T_target = torch.tensor(target_temperature, device='cuda', dtype=torch.float32)
        self.T_Grad_target = torch.tensor(T_Grad_target, device='cuda', dtype=torch.float32)
        self.N = len(time_step_l)
        self.iterations = iterations
        self.hidden_dim = hidden_dim
        # 初始化时间矩阵序列
        self.time_D_m_base = (self.time_step_l / time_D_max).view(1, self.N, 1, 1, 1)

    def objective(self, u_plan):
        # 内存隔离
        m_curr_step = self.m_curr.clone()
        h_old_step = self.h_old.clone()
        # 创建风扇配置矩阵序列
        cooling_m_seq = differentiable_cooling_matrix(u_plan, global_mask1, global_mask2, testsize)

        total_heat_cost = torch.tensor(0.0, device='cuda', dtype=torch.float32)
        total_grad_cost = torch.tensor(0.0, device='cuda', dtype=torch.float32)
        total_cooling_cost = torch.tensor(0.0, device='cuda', dtype=torch.float32)
        # 超温损失的累积计算
        for i in range(self.N):
            time_D_m = self.time_D_m_base[:, i, :, :, :]       # 这种读取方式会自动压缩一个通道
            current_cooling_m = cooling_m_seq[:, i, :, :, :]   # 这种读取方式会自动压缩一个通道
            m_next_pred, h_curr_next = transient_model(
                time_D_m, m_curr_step, self.flux_m, current_cooling_m, self.envtemp_m, h_old_step
            )
            m_old_step = m_curr_step 
            m_curr_step = m_next_pred
            h_old_step = h_curr_next
            # ------------------------------------------------- #
            # 温度场的超温损失
            time_weight = torch.sqrt(torch.tensor(
                (i + 1) / self.N, device=m_curr_step.device, dtype=m_curr_step.dtype
            ))
            heat_error = torch.mean(torch.clamp(m_curr_step - self.T_target, min=0.0))
            step_heat_cost = heat_epsilon * (
                torch.sqrt(1.0 + (heat_error / heat_epsilon)**2) - 1.0
            )
            # 温度场的梯度损失
            grad_error = torch.mean(torch.clamp(
                (m_curr_step - m_old_step) / self.time_step_l[:, i, :]
                - self.T_Grad_target,
                min=0.0
            ))
            step_grad_cost = grad_epsilon * (
                torch.sqrt(1.0 + (grad_error / grad_epsilon)**2) - 1.0
            )
            # 非均匀权重配置
            total_heat_cost = total_heat_cost + time_weight * step_heat_cost / self.N
            total_grad_cost = total_grad_cost + time_weight * step_grad_cost / self.N
            # 冷却耗能损失
            u_i = u_plan[:, i, :]
            total_cooling_cost = total_cooling_cost + torch.mean(u_i.pow(3)).pow(1.0 / 3.0) / self.N

        # Heat Cost 和 Cooling Cost 的流动路径不同，单看 cost 数量级不可比
        return total_heat_cost.mean(), total_grad_cost.mean(), total_cooling_cost

    def optimize(self, m_curr, u_init, h_old):
        self.m_curr = m_curr.detach()
        b, _, h, w = self.m_curr.shape
        if h_old is None:
            self.h_old = torch.zeros((b, self.hidden_dim, h, w), device=self.m_curr.device)
        else:
            self.h_old = h_old.detach()

        self.last_heat_grad = None
        self.last_grad_grad = None
        self.last_cooling_grad = None

        def objective_with_grad(u_numpy):
            u_plan = torch.tensor(
                u_numpy, device='cuda', dtype=torch.float32
            ).view(1, 2).requires_grad_(True)
            u_plan_expanded = u_plan.unsqueeze(1).expand(-1, self.N, -1)
            total_heat_cost, total_grad_cost, total_cooling_cost = self.objective(u_plan_expanded)
            total_heat_cost = weight_heat * total_heat_cost
            total_grad_cost = weight_grad * total_grad_cost
            total_cooling_cost = weight_cool * total_cooling_cost

            self.last_heat_grad = torch.autograd.grad(
                total_heat_cost, u_plan, retain_graph=True
            )[0].detach()
            self.last_grad_grad = torch.autograd.grad(
                total_grad_cost, u_plan, retain_graph=True
            )[0].detach()
            self.last_cooling_grad = torch.autograd.grad(
                total_cooling_cost, u_plan, retain_graph=True
            )[0].detach()
            cost = total_heat_cost + total_grad_cost + total_cooling_cost
            cost_grad = torch.autograd.grad(cost, u_plan)[0]
            return cost.item(), cost_grad.detach().cpu().numpy().ravel().astype(np.float64)

        u_min = self.u_min.item()
        u_max = self.u_max.item()
        result = minimize(
            objective_with_grad,
            u_init[:, 0, :].detach().cpu().numpy().ravel().astype(np.float64),
            method='SLSQP',
            jac=True,
            bounds=[(u_min, u_max), (u_min, u_max)],
            options={'maxiter': self.iterations, 'ftol': optimizer_ftol, 'disp': False},
        )
        u_plan = torch.tensor(result.x, device='cuda', dtype=torch.float32).view(1, 2)
        u_plan_expanded = u_plan.unsqueeze(1).expand(-1, self.N, -1)
        return u_plan_expanded.clone().detach(), result.nit, \
               self.last_heat_grad, self.last_grad_grad, self.last_cooling_grad

fan_p_accu = 0
time_step_list = []
if __name__ == '__main__':
    numpyfile = os.path.join(savepath, 'optimization_numpy_file_ttc')
    plotfile = os.path.join(savepath, 'optimization_plot_file_ttc')
    os.makedirs(numpyfile, exist_ok=True)
    os.makedirs(plotfile, exist_ok=True)
    print('========== Fixed configuration ==========' )

    temp_scale = temp_max - temp_min
    target_temperature = (target_temp - temp_min) / temp_scale
    environment_temperature = (env_temp - temp_min) / temp_scale
    initial_temperature = (initial_temp - temp_min) / temp_scale

    np.save(os.path.join(numpyfile, 'target_temperature.npy'), np.array(target_temperature))
    np.save(os.path.join(numpyfile, 'environment_temperature.npy'), np.array(environment_temperature))
    np.save(os.path.join(numpyfile, 'initial_temperature.npy'), np.array(initial_temperature))
    np.save(os.path.join(numpyfile, 'chip_power.npy'), np.array(chip_power))

    print(f'Target temperature: {target_temp:.1f} C ({target_temperature:.6f})')
    print(f'Environment temperature: {env_temp:.1f} C ({environment_temperature:.6f})')
    print(f'Initial temperature: {initial_temp:.1f} C ({initial_temperature:.6f})')

    # 初始化
    flux_m = prepare_flux_matrix(chip_loc, chip_power, testsize).cuda()
    m_curr = torch.full(
        (1, 1, testsize[0], testsize[1]),
        initial_temperature,
        dtype=torch.float32,
        device='cuda',
    )
    m_curr_MPC = m_curr.clone()
    envtemp_m = torch.full(
        (1, 1, testsize[0], testsize[1]),
        environment_temperature,
        dtype=torch.float32,
        device='cuda',
    )
    print(
        'Initial field range: '
        f'{m_curr.min().item() * temp_scale + temp_min:.3f} - '
        f'{m_curr.max().item() * temp_scale + temp_min:.3f} C'
    )

    # 使用当前热源功率，在最小风扇转速下计算首步最大升温速率
    fan_min_normalized = (fan_min - c_min) / (c_max - c_min)
    u_grad_reference = torch.full(
        (1, 1, 2),
        fan_min_normalized,
        dtype=torch.float32,
        device='cuda',
    )
    cooling_grad_reference = differentiable_cooling_matrix(
        u_grad_reference,
        global_mask1,
        global_mask2,
        testsize,
    ).squeeze(1)
    time_D_grad_reference = torch.full(
        (1, 1, testsize[0], testsize[1]),
        time_step / time_D_max,
        dtype=torch.float32,
        device='cuda',
    )
    with torch.no_grad():
        m_next_grad_reference, _ = transient_model(
            time_D_grad_reference,
            m_curr,
            flux_m,
            cooling_grad_reference,
            envtemp_m,
            None,
        )

    first_step_max_rate = torch.max(
        (m_next_grad_reference - m_curr) / time_step
    ).item()
    T_Grad_target = T_Grad_ratio * first_step_max_rate
    first_step_max_rate_C_per_s = first_step_max_rate * temp_scale
    T_Grad_target_C_per_s = T_Grad_target * temp_scale

    np.save(os.path.join(numpyfile, 'T_Grad_target.npy'), np.array(T_Grad_target))
    np.save(
        os.path.join(numpyfile, 'first_step_max_rate_C_per_s.npy'),
        np.array(first_step_max_rate_C_per_s),
    )
    print(
        f'Reference first-step max rate at [{fan_min:g}, {fan_min:g}] RPM: '
        f'{first_step_max_rate_C_per_s:.6e} C/s'
    )
    print(
        f'T Gradient target ({T_Grad_ratio:g} x reference): '
        f'{T_Grad_target_C_per_s:.6e} C/s '
        f'({T_Grad_target:.6e} normalized/s)'
    )

    h_old = None
    h_old_MPC = None
    pred_curves = []
    pred_curves_MPC = []
    time_seq = []
    time_step_list = []
    iteration_list = []

    heat_grad_u1_list = []
    heat_grad_u2_list = []
    grad_grad_u1_list = []
    grad_grad_u2_list = []
    cooling_grad_u1_list = []
    cooling_grad_u2_list = []

    time_accum = 0   # 当前的时间
    s = 0   # 当前的时间步数
    # 初始配置
    N = len(time_step_l)
    # 冷启动
    u_init = ((torch.full((1, N, 2), fan_start, dtype=torch.float32, device='cuda') - c_min) / (c_max - c_min)).cuda()
    # 初始化
    # 风扇
    Fan_1_seq = []
    Fan_2_seq = []
    # 时间记录
    time_seq = []

    u_non = u_init
    # 声明优化引擎
    optimizer_engine = MPC_Optimizer(
        flux_m=flux_m,            # 离线热设计优化不变量 (在线热控制变量)
        envtemp_m=envtemp_m,      # 离线热设计优化不变量 (在线热控制变量)
        target_temperature=target_temperature,
        T_Grad_target=T_Grad_target,
        time_step_l=time_step_l,  # 离线热设计优化不变量 (在线热控制变量)
        iterations=optimizer_iterations)

    while time_accum < time_total:
        s += 1
        if h_old_MPC is not None:
            u_init = u_best_sequence
        time_accum += time_step
        time_seq.append(time_accum)
        # 生成 MPC 预测所需的时间步长矩阵
        time_D_m = torch.tensor(time_step/time_D_max).view(1, 1, 1, 1).expand(1, 1, testsize[0], testsize[1]).cuda()
        print(f'# -------- Step {s} -------- #')
        # ----------------------------------------------- #
        # # === 不优化曲线===
        with torch.no_grad():
            cooling_m = differentiable_cooling_matrix(u_non[:1, :1, :], 
                                                      global_mask1, 
                                                      global_mask2, 
                                                      testsize).squeeze(1)
            m_next_pred, h_curr = transient_model(time_D_m, m_curr, flux_m, cooling_m.cuda(), envtemp_m, h_old)
        m_curr = m_next_pred.detach()
        h_old = h_curr.detach()
        # === 数据后处理 ===
        pred_points = chip_temp_detect(m_curr, chip_loc, testsize)
        pred_curves.append(pred_points)
        # ----------------------------------------------- #
        # MPC优化器
        # 梯度优化器
        optimization_start = time.time()
        # 迭代优化
        u_best_sequence, iters, last_heat_grad, last_grad_grad, last_cooling_grad = optimizer_engine.optimize(
            m_curr=m_curr_MPC,   # 离线热设计优化变量
            u_init=u_init,      # 离线热设计优化变量
            h_old=h_old_MPC,     # 离线热设计优化变量
            )
        optimization_time = time.time() - optimization_start
        time_step_list.append(optimization_time)
        iteration_list.append(iters)

        heat_grad_u1_list.append(last_heat_grad[:,0].item())
        heat_grad_u2_list.append(last_heat_grad[:,1].item())
        grad_grad_u1_list.append(last_grad_grad[:,0].item())
        grad_grad_u2_list.append(last_grad_grad[:,1].item())
        cooling_grad_u1_list.append(last_cooling_grad[:,0].item())
        cooling_grad_u2_list.append(last_cooling_grad[:,1].item())

        # 截取第一步风扇动作并转换为 Tensor 供前向演化使用
        u_best_sequence_cut = np.squeeze(u_best_sequence.cpu().numpy(), axis=0)
        Fan_1_seq.append((c_max - c_min) * u_best_sequence_cut[0][0] + c_min)
        Fan_2_seq.append((c_max - c_min) * u_best_sequence_cut[0][1] + c_min)

        u_act = torch.tensor(u_best_sequence_cut[0]).unsqueeze(0).unsqueeze(0).cuda()
        
        # --- 真实演化一步：必须使用 torch.no_grad() 防止显存泄漏 ---
        with torch.no_grad():
            cooling_m_MPC = differentiable_cooling_matrix(u_act, global_mask1, global_mask2, testsize).squeeze(1)
            m_next_pred_MPC, h_curr_MPC = transient_model(
                time_D_m, m_curr_MPC, flux_m, cooling_m_MPC, envtemp_m, h_old_MPC
            )
            
        m_curr_MPC = m_next_pred_MPC.detach()
        h_old_MPC = h_curr_MPC.detach()
        # ----------------------------------------------- #
        # === 数据后处理 ===
        pred_points = chip_temp_detect(m_curr_MPC, chip_loc, testsize)
        pred_curves_MPC.append(pred_points)
        # ----------------------------------------------- #
        print(f"Step {s} optimized in {optimization_time:.2f} seconds, and took {iters} optimization iterations.")
        print(f"Time accumulated: {time_accum}")

        # 结果可视化
        pred_curves_plot = np.array(pred_curves)
        pred_curves_MPC_plot = np.array(pred_curves_MPC)
        fan_1_np = np.array(Fan_1_seq)
        fan_2_np = np.array(Fan_2_seq)
        x_axis_data = np.array(time_seq)

        heat_grad_u1_plot = np.array(heat_grad_u1_list)
        heat_grad_u2_plot = np.array(heat_grad_u2_list)
        grad_grad_u1_plot = np.array(grad_grad_u1_list)
        grad_grad_u2_plot = np.array(grad_grad_u2_list)
        cooling_grad_u1_plot = np.array(cooling_grad_u1_list)
        cooling_grad_u2_plot = np.array(cooling_grad_u2_list)
        optimization_time_plot = np.array(time_step_list)
        iteration_plot = np.array(iteration_list)

        if s % save_interval == 0: # 每隔固定步数保存一次图像
            fig, axes = plt.subplots(2, 6, figsize=(20, 6))
            axes = axes.flatten()
            for i in range(12): # 12个芯片
                ax = axes[i]
                ax.plot(x_axis_data[:s], pred_curves_plot[:, i], 'b--', label='Pred')
                ax.plot(x_axis_data[:s], pred_curves_MPC_plot[:, i], 'r--', label='Pred_SLSQP')
                ax.axhline(target_temperature, color='k', linestyle=':', linewidth=1.0, label='Target')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                if i == 0: ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(plotfile, f'temperature_curves_{s}.png'), dpi=150)
            plt.close()

            # 确保转换为 NumPy 数组
            fig, ax = plt.subplots(figsize=(10, 5))
            # 绘制曲线，设置不同的颜色和线型
            ax.plot(x_axis_data[:s], fan_1_np, label='Fan 1 Speed', color='#1f77b4', linewidth=2, linestyle='-')
            ax.plot(x_axis_data[:s], fan_2_np, label='Fan 2 Speed', color='#d62728', linewidth=2, linestyle='--')
            # 图表装饰
            ax.set_title('Fan Speed Control Sequence over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Fan Speed (RPM)', fontsize=12)
            # 设置合理的 Y 轴范围 (根据你的实际限幅修改，比如 1000 到 6300)
            ax.set_ylim(800, 6500) 
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            ax.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plotfile, f'fan_curves_{s}.png'), dpi=150)
            plt.close()


            # 确保转换为 NumPy 数组
            fig, ax = plt.subplots(2, 1, figsize=(6, 8))
            # 绘制曲线，设置不同的颜色和线型
            ax[0].plot(x_axis_data[:s], heat_grad_u1_plot, label='cost heat grad u1', color='#1f77b4', linewidth=2, linestyle='-')
            ax[0].plot(x_axis_data[:s], grad_grad_u1_plot, label='cost grad grad u1', color='#d62728', linewidth=2, linestyle='--')
            ax[0].plot(x_axis_data[:s], cooling_grad_u1_plot, label='cost cooling grad u1', color='#2ca02c', linewidth=2, linestyle=':')

            ax[1].plot(x_axis_data[:s], heat_grad_u2_plot, label='cost heat grad u2', color='#1f77b4', linewidth=2, linestyle='-')
            ax[1].plot(x_axis_data[:s], grad_grad_u2_plot, label='cost grad grad u2', color='#d62728', linewidth=2, linestyle='--')
            ax[1].plot(x_axis_data[:s], cooling_grad_u2_plot, label='cost cooling grad u2', color='#2ca02c', linewidth=2, linestyle=':')
            # 图表装饰
            ax[0].set_title('Gradient Sequence over Time', fontsize=14, fontweight='bold')
            ax[0].set_xlabel('Time Step', fontsize=12)
            ax[1].set_title('Gradient Sequence over Time', fontsize=14, fontweight='bold')
            ax[1].set_xlabel('Time Step', fontsize=12)
            # ax.set_ylabel('Fan Speed (RPM)', fontsize=12)
            # 设置合理的 Y 轴范围 (根据你的实际限幅修改，比如 1000 到 6300)
            ax[0].grid(True, which='both', linestyle='--', alpha=0.6)
            ax[1].grid(True, which='both', linestyle='--', alpha=0.6)
            ax[0].legend(loc='upper right', fontsize=10)
            ax[1].legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(plotfile, f'Grad_curves_{s}.png'), dpi=150)
            plt.close()

            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(x_axis_data[:s], optimization_time_plot, color='#2ca02c', linewidth=1.5)
            ax[0].set_title('SLSQP Optimization Performance', fontsize=14, fontweight='bold')
            ax[0].set_ylabel('Optimization Time (s)', fontsize=12)
            ax[0].grid(True, which='both', linestyle='--', alpha=0.6)

            ax[1].plot(x_axis_data[:s], iteration_plot, color='#9467bd', linewidth=1.5)
            ax[1].axhline(optimizer_iterations, color='#d62728', linestyle='--', linewidth=1.5,
                          label=f'Max Iterations = {optimizer_iterations}')
            ax[1].set_xlabel('Time (s)', fontsize=12)
            ax[1].set_ylabel('Iterations', fontsize=12)
            ax[1].set_ylim(0, optimizer_iterations + 1)
            ax[1].grid(True, which='both', linestyle='--', alpha=0.6)
            ax[1].legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(plotfile, f'optimization_performance_{s}.png'), dpi=150)
            plt.close()


            # 保存为numpy数组 
            time_list_np = np.array(time_step_list)
            np.save(os.path.join(numpyfile, f'time_seq_{s}.npy'), x_axis_data)
            np.save(os.path.join(numpyfile, f'time_step_list_{s}.npy'), time_list_np)
            np.save(os.path.join(numpyfile, f'iteration_list_{s}.npy'), iteration_plot)
            np.save(os.path.join(numpyfile, f'fan_1_{s}.npy'), fan_1_np)
            np.save(os.path.join(numpyfile, f'fan_2_{s}.npy'), fan_2_np)
            np.save(os.path.join(numpyfile, f'temp_chips_original_{s}.npy'), pred_curves_plot)
            np.save(os.path.join(numpyfile, f'temp_chips_MPC_{s}.npy'), pred_curves_MPC_plot)

            np.save(os.path.join(numpyfile, f'heat_grad_u1_{s}.npy'), np.array(heat_grad_u1_list))
            np.save(os.path.join(numpyfile, f'heat_grad_u2_{s}.npy'), np.array(heat_grad_u2_list))
            np.save(os.path.join(numpyfile, f'grad_grad_u1_{s}.npy'), np.array(grad_grad_u1_list))
            np.save(os.path.join(numpyfile, f'grad_grad_u2_{s}.npy'), np.array(grad_grad_u2_list))
            np.save(os.path.join(numpyfile, f'cooling_grad_u1_{s}.npy'), np.array(cooling_grad_u1_list))
            np.save(os.path.join(numpyfile, f'cooling_grad_u2_{s}.npy'), np.array(cooling_grad_u2_list))
            # ================= 2. 自动清理旧文件机制 ================= #
            prev_s = s - save_interval
            if prev_s > 0:
                # 图像文件路径
                files_to_delete = [
                    os.path.join(plotfile, f'temperature_curves_{prev_s}.png'),
                    os.path.join(plotfile, f'fan_curves_{prev_s}.png'),
                    os.path.join(plotfile, f'Grad_curves_{prev_s}.png'),
                    os.path.join(plotfile, f'optimization_performance_{prev_s}.png'),
                    os.path.join(numpyfile, f'time_seq_{prev_s}.npy'),
                    os.path.join(numpyfile, f'time_step_list_{prev_s}.npy'),
                    os.path.join(numpyfile, f'iteration_list_{prev_s}.npy'),
                    os.path.join(numpyfile, f'fan_1_{prev_s}.npy'),
                    os.path.join(numpyfile, f'fan_2_{prev_s}.npy'),
                    os.path.join(numpyfile, f'temp_chips_original_{prev_s}.npy'),
                    os.path.join(numpyfile, f'temp_chips_MPC_{prev_s}.npy'),
                    os.path.join(numpyfile, f'heat_grad_u1_{prev_s}.npy'),
                    os.path.join(numpyfile, f'heat_grad_u2_{prev_s}.npy'),
                    os.path.join(numpyfile, f'grad_grad_u1_{prev_s}.npy'),
                    os.path.join(numpyfile, f'grad_grad_u2_{prev_s}.npy'),
                    os.path.join(numpyfile, f'cooling_grad_u1_{prev_s}.npy'),
                    os.path.join(numpyfile, f'cooling_grad_u2_{prev_s}.npy')
                ]
                
                # 安全删除
                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"清理旧文件失败: {file_path}, 错误: {e}")
