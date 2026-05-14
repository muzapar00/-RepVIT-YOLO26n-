"""
RepViT-C3k2: RepViTBlock + C3k2 融合模块
==========================================
论文创新：将 RepViT (CVPR 2024) 的重参数化高效 Mobile-Style Block
与 YOLO26 的 C3k2 CSP Bottleneck 进行结构融合。

设计思路:
  - C3k2 框架不变（CSP 分支 + 多 Bottleneck 串联）
  - 将内部 Bottleneck 替换为 RepViTBlock（重参数化 token mixer + channel mixer）
  - RepViTBlock 训练时使用多分支（DWConv7x7 + DWConv5x5 + DWConv3x3 + Identity），
    推理时融合为单分支，零额外开销
  - 加入 SE 注意力增强通道特征筛选

接口兼容 ultralytics YOLO26 的模块注册机制，
通过 yaml 配置文件中的 "RepViTBlock" 关键词自动加载。

参数说明 (从 YAML 读取):
  - c2: 输出通道数（会根据 width 缩放）
  - n: RepViTBlock 重复次数
  - shortcut: 是否使用跨层连接
  - e: CSP 扩展比例
  - kernel_size: RepViTBlock token mixer 卷积核大小

示例 YAML (YOLO26n):
  [-1, 2, RepViTC3k2, [256, 1, False, 0.5, 7]]
  #    c2=256  n=1  shortcut=False  e=0.5  kernel_size=7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RepViTBlock", "RepViTC3k2"]


class RepViTBlock(nn.Module):
    """RepViT Block: 重参数化的轻量级特征提取单元.

    结构 (训练时):
        x ──┬── DWConv 7x7 ──┐
            ├── DWConv 5x5 ──┤── BN ── ReLU ── x_out
            ├── DWConv 3x3 ──┤
            └── Identity  ───┘

    推理时: 4 个分支融合为单个 DWConv 7x7，实现零额外推理开销.
    之后接 Channel Mixer (Pointwise-Expand -> DWConv -> Pointwise-Project).

    参考:
        Wang et al., "RepViT: Revisiting Mobile CNN From ViT Perspective", CVPR 2024.
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        """
        Args:
            dim: 输入/输出通道数.
            kernel_size: 训练时最大的 token mixer 卷积核大小 (默认 7).
        """
        super().__init__()
        # ── Token Mixer (多分支重参数化深度卷积) ──
        padding = kernel_size // 2
        self.token_mixer_dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False
        )
        self.token_mixer_bn = nn.BatchNorm2d(dim)

        # 辅助分支 (训练用，推理时融合进主分支)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        # Identity 分支 BN
        self.bn_identity = nn.BatchNorm2d(dim)

        self.act = nn.ReLU(inplace=True)

        # ── Channel Mixer (倒残差 FFN) ──
        hidden = int(dim * 4)
        self.channel_mixer = nn.Sequential(
            # Expand
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            # Project
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # ── SE 注意力 ──
        se_dim = max(dim // 4, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, se_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 (训练模式: 多分支; 推理模式: 融合后单分支)."""
        # Token Mixer
        if hasattr(self, "dwconv1"):
            # 训练模式: 多分支
            out = (
                self.token_mixer_bn(self.token_mixer_dwconv(x))
                + self.bn1(self.dwconv1(x))
                + self.bn2(self.dwconv2(x))
                + self.bn_identity(x)
            )
        else:
            # 推理模式: 已融合的单分支
            out = self.token_mixer_bn(self.token_mixer_dwconv(x))

        out = self.act(out)

        # SE 注意力
        se_weight = self.se(out)
        out = out * se_weight

        # Channel Mixer
        out = self.channel_mixer(out) + x  # 残差连接
        return out

    @torch.no_grad()
    def repvgg_reparameterize(self):
        """将多分支 Token Mixer 融合为单个 DWConv (推理加速).

        融合规则:
            W_fused = W_7x7 + pad(W_5x5, 2) + pad(W_3x3, 2) + W_identity
            BN_fused = fuse(BN_7x7, BN_5x5, BN_3x3, BN_identity)
        """
        if not hasattr(self, "dwconv1"):
            return  # 已经融合过

        # 融合 BN
        def _fuse_conv_bn(conv, bn):
            """将 Conv + BN 融合为带 bias 的 Conv."""
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        # 各分支的权重和偏置
        k7, b7 = _fuse_conv_bn(self.token_mixer_dwconv, self.token_mixer_bn)
        k5, b5 = _fuse_conv_bn(self.dwconv2, self.bn2)
        k3, b3 = _fuse_conv_bn(self.dwconv1, self.bn1)

        # Identity 分支
        dim = self.bn_identity.num_features
        k_id = torch.zeros_like(k7)
        for i in range(dim):
            k_id[i, i % (dim // self.token_mixer_dwconv.groups),
                k_id.shape[2] // 2, k_id.shape[3] // 2] = 1.0
        _, b_id = _fuse_conv_bn(
            nn.Conv2d(dim, dim, 1, groups=1, bias=False), self.bn_identity
        )

        # 将 5x5 和 3x3 padding 到 7x7
        k5_padded = F.pad(k5, [1, 1, 1, 1])
        k3_padded = F.pad(k3, [2, 2, 2, 2])

        # 融合所有分支
        k_fused = k7 + k5_padded + k3_padded + k_id
        b_fused = b7 + b5 + b3 + b_id

        # 创建融合后的单分支
        fused_conv = nn.Conv2d(
            dim, dim,
            kernel_size=self.token_mixer_dwconv.kernel_size,
            padding=self.token_mixer_dwconv.padding,
            groups=self.token_mixer_dwconv.groups,
            bias=True,
        ).to(k_fused.device)
        fused_conv.weight.data = k_fused
        fused_conv.bias.data = b_fused

        # 替换
        self.token_mixer_dwconv = fused_conv
        self.token_mixer_bn = nn.Identity()

        # 删除辅助分支
        del self.dwconv1, self.dwconv2, self.bn1, self.bn2, self.bn_identity


class RepViTC3k2(nn.Module):
    """RepViT-C3k2: 将 C3k2 的内部 Bottleneck 替换为 RepViTBlock.

    继承 C2f 的 CSP 框架:
        输入 -> cv1(1x1) -> split -> [RepViTBlock x n] -> concat -> cv2(1x1) -> 输出

    与标准 C3k2 的区别:
        - 内部模块从 Bottleneck/C3k 替换为 RepViTBlock
        - RepViTBlock 使用重参数化大核深度卷积作为 token mixer
        - 加入 SE 注意力增强通道特征
        - 推理时可融合为单分支，无额外开销
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        kernel_size: int = 7,
        shortcut: bool = True,
        g: int = 1,
    ):
        """
        Args:
            c1: 输入通道数.
            c2: 输出通道数.
            n: RepViTBlock 重复次数.
            c3k: 保留参数（兼容性，此处不使用）.
            e: CSP 扩展比例.
            kernel_size: RepViTBlock 的 token mixer 卷积核大小.
            shortcut: 是否使用跨层连接 (CSP 框架自带).
            g: 卷积分组数.
        """
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        # 确保 n 是整数
        n = int(n) if n > 0 else 1

        # 确保 e 是合理的范围
        if not (0 < e <= 1.0):
            e = 0.5

        # 确保 shortcut 是 bool 类型（YAML 中可能是 "False"/"True" 字符串或 bool）
        shortcut = bool(shortcut)

        # 确保 kernel_size 是整数
        kernel_size = int(kernel_size) if kernel_size > 0 else 7

        # 调试输出
        import sys
        print(f"[RepViTC3k2] 收到的参数: c1={c1}, c2={c2}, n={n}, e={e}, kernel_size={kernel_size}, shortcut={shortcut}")

        self.c = int(c2 * e)  # hidden channels
        print(f"[RepViTC3k2] self.c = int(c2 * e) = int({c2} * {e}) = {self.c}")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RepViTBlock(self.c, kernel_size=kernel_size)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
