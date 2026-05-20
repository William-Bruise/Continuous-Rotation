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
# 默认仅检查/下载DIV2K训练数据，不自动下载benchmark测试集
```

若你不希望看到网络告警，可只建目录：
```bash
python scripts/prepare_data.py --root ./data
# 默认仅检查/下载DIV2K训练数据，不自动下载benchmark测试集 --no-download
```

若你在CI里希望下载失败时返回非0退出码：
```bash
python scripts/prepare_data.py --root ./data
# 默认仅检查/下载DIV2K训练数据，不自动下载benchmark测试集 --strict-download
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
如果 `prepare_data.py` 里 `benchmark_report.status=warning_unavailable_network`，通常是网络或镜像不可达，不是代码崩溃。当你显式开启 `--download-benchmarks` 时，脚本会尝试官方 EDSR benchmark 链接；若失败，请手动放置到：
- `data/benchmarks/Set5/HR`
- `data/benchmarks/Set14/HR`
- `data/benchmarks/BSD100/HR`
- `data/benchmarks/Urban100/HR`

之后重新运行 `prepare_data.py`，应看到 benchmark 图片计数大于 0。


### 可选：下载测试基准集（仅测试用）
```bash
python scripts/prepare_data.py --root ./data --download-benchmarks
```
说明：训练只需要 DIV2K。Set5/Set14/BSD100/Urban100 只用于测试。


## 本地离线压缩包放置位置（你这种情况）
如果服务器无法联网下载，请把压缩包放到：
- `data/_downloads/Set5.zip`
- `data/_downloads/Set14.zip`
- `data/_downloads/BSD100.zip`
- `data/_downloads/Urban100.zip`
（也支持 `data/_downloads/benchmark.tar` 或 `benchmark.zip`）

然后运行：
```bash
python scripts/prepare_data.py --root ./data --download-benchmarks
```
脚本会自动解压并拷贝到：
- `data/benchmarks/Set5/HR`
- `data/benchmarks/Set14/HR`
- `data/benchmarks/BSD100/HR`
- `data/benchmarks/Urban100/HR`


## 论文设定对齐训练配置
默认 `configs/default_sr.yaml` 已对齐为：
- DIV2K 训练
- scale range `[2,4]`
- `batch_size=16`
- `num_epochs=1000`
- Adam 初始学习率 `1e-4`
- 每 `200` epoch 学习率衰减到 `0.5x`
- benchmark 测试输出 `benchmark_table.json/csv`，包含 Set5/Set14/BSD100/Urban100 在 in-scale(x2/x3/x4) 与 out-scale(x6/x8/x12) 的 PSNR/SSIM


## 参数量自动统计
运行：
```bash
python scripts/count_params.py --config configs/default_sr.yaml --out outputs/param_count.json
```
会输出：
- 总参数量（`total_params` / `total_params_million`）
- 各子模块参数量（`module_params`）

用于和论文表格 `Param.` 列直接对齐。


## 参数量对齐建议（与你表格的 1.3M 档位）
当前 `continuous_so2` 的参数主要由 `coeff_encoder` 决定。
若 `scripts/count_params.py` 输出过小，可优先调大 `model.encoder_channels`。
默认配置已上调到 `encoder_channels: 80`，用于接近 1.3M 量级。
推荐用以下命令复查：
```bash
python scripts/count_params.py --config configs/default_sr.yaml --out outputs/param_count.json
```
