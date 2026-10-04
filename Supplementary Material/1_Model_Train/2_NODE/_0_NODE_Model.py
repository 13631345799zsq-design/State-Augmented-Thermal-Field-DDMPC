import torch
import torch.nn as nn

# ==========================================
# 2. CNN-NODE 微分方程
# ==========================================
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        in_channels = 3
        out_channels = 1
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(16, 128), nn.SiLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True))
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True))
        # Decoder
        self.up4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(32, 256), nn.SiLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(384, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(16, 128), nn.SiLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(inplace=True))
        # Final Output
        self.final = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=3, padding=1))

    def forward(self, state, flux_m, cooling_m):
        temp_m_curr = state
        x = torch.cat([temp_m_curr, flux_m, cooling_m], dim=1)
        # 1. Encoding
        d1 = self.down1(x)      # [64, H/2, W/2]
        d2 = self.down2(d1)     # [128, H/4, W/4]
        d3 = self.down3(d2)     # [256, H/8, W/8]
        d4 = self.down4(d3)     # [512, H/16, W/16]
        # 2. Bottleneck
        b = self.bottleneck(d4) # [512, H/16, W/16]
        # 3. Decoding (Explicit Skip Connections)
        u4 = self.up4(torch.cat((b, d4), 1))
        u3 = self.up3(torch.cat((u4, d3), 1))
        u2 = self.up2(torch.cat((u3, d2), 1))
        u1 = self.up1(torch.cat((u2, d1), 1))
        output = self.final(u1)
        return output

# ==========================================
# 3. 顶层物理约束 CNN-NODE
# ==========================================
class PhysicsInformedCNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedCNN, self).__init__()
        self.ode_func = ODEFunc()

    def forward(self, time_D_m, temp_m_curr, flux_m, cooling_m, envtemp_m):
        theta_m_curr = temp_m_curr - envtemp_m
        alpha = self.ode_func(theta_m_curr, flux_m, cooling_m)
        temp_m_curr_next = temp_m_curr + alpha * time_D_m
        return temp_m_curr_next
