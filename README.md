# Continuous-Rotation SR Prototype

## 状态声明（重要）
- **当前版本已实现“连续角变量主导”的 SO(2) 近似建模**：orientation 用连续 `θ∈[0,2π)` 查询，不再以 rotation bin 作为主表示对象。
- **仍非严格数学上完全连续等变**：因为仍有数值近似（有限 Fourier 阶数、有限 quadrature、grid_sample 插值、encoder 非严格 steerable）。

## 连续目标 vs 当前实现
1. 理论目标：continuous SO(2)/O(2)-equivariant restoration。  
2. 表示层实现：**Fourier coefficient field**（`A0, Acos, Asin`）定义连续 `z(u,θ)`。  
3. 数值近似：空间 bilinear 采样 + 有限 Fourier truncation `M` + 有限 quadrature `Q` + 图像旋转插值。

## 新增连续表示核心
- `models/continuous.py`
  - `FourierCoeffEncoder`: 输出 `A0∈R[B,C,H,W]`, `Acos/Asin∈R[B,M,C,H,W]`
  - `sample_orientation_feature(coeffs, coords, theta)`: 实现
    `z(p,θ)=a0(p)+Σ_m[a_m(p)cos(mθ)+b_m(p)sin(mθ)`]
  - `OrientationQuadratureAggregator`: 用 `Q` 个 `θ_q` 做数值积分近似
    `q(p)≈(1/Q)Σ_q z(p,θ_q)`
  - `ContinuousGroupO2LIIFSR`: `CoeffEncoder -> ContinuousSampler -> QuadratureAgg -> Decoder`

## 连续等变损失与指标
- 损失：`continuous_equivariance_loss`
  - `θ~Uniform(0,2π)`
  - `L_eq = || f(T_θ y, p) - f(y, R_{-θ}p) ||_1`
- 指标：
  - `rot_ee`（离散角列表）
  - `rot_ee_cont`（连续随机角采样）

## 运行方式
### 1) 准备数据
```bash
python scripts/prepare_data.py --root ./data
```

### 2) 训练（默认 continuous_so2）
```bash
python scripts/train_sr.py --config configs/default_sr.yaml
```

### 3) 测试
```bash
python scripts/test_sr.py --config configs/default_sr.yaml --ckpt outputs/<exp>/checkpoints/best.pt --out outputs/test_eval
```

### 4) 可选谱扫描（legacy离散谱模块）
```bash
python scripts/scan_spectral.py --config configs/default_sr.yaml --out outputs/spectral_scan.json
```

## 关键配置
`configs/default_sr.yaml` 里可调：
- `model.num_fourier_orders` (M)
- `model.num_orientation_quadrature` (Q)
- `model.orientation_quadrature_mode` (`uniform`/`random`)
- `loss.use_continuous_eq_loss`
- `loss.continuous_eq_loss_weight`
- `eval.num_continuous_eval_angles`
- `model.max_rotation_radians`
- `model.align_corners`
- `model.rotation_interp_mode`

## Legacy说明
- `GroupO2LIIFSR` / `models/spectral.py` / lifting-by-bins 路径仍保留用于对照实验，属于 legacy discrete approximation。


## 下载失败排查（例如 Connection reset by peer）
如果 `prepare_data.py` 里 `benchmark_report.status=unavailable_network`，通常是网络或镜像不可达，不是代码崩溃。脚本已自动尝试多个URL；若都失败，请手动放置到：
- `data/benchmarks/Set5/HR`
- `data/benchmarks/Set14/HR`
- `data/benchmarks/BSD100/HR`
- `data/benchmarks/Urban100/HR`

之后重新运行 `prepare_data.py`，应看到 benchmark 图片计数大于 0。
