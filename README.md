# Continuous-Rotation: O2LIIFSR Prototype

## 1. 项目目标
实现 arbitrary-scale implicit SR 原型，并加入旋转/反射一致性正则与评估，为未来严格 O(2)-equivariant 版本铺路。

## 2. 当前实现范围
- LIIF-style pipeline: Encoder + Local sampler + Implicit decoder
- 训练/验证/测试一键流程
- Rotation/Reflection consistency loss（soft regularization）
- PSNR/SSIM/Rot-EE/Ref-EE 指标
- EquivariantEncoder / O2ImplicitDecoder 占位接口

## 3. 如何准备数据
```bash
python scripts/prepare_data.py --root ./data
```
脚本会：
- 自动创建 `train_hr/val_hr/test_hr`
- 自动尝试下载 DIV2K train/valid HR（若公网可达）
- 若下载失败，会保留目录并给出 `download_report`，你仍可手动放置图像

仅创建目录（不尝试下载）：
```bash
python scripts/prepare_data.py --root ./data --no-download
```

## 4. 如何训练
```bash
python scripts/train_sr.py --config configs/default_sr.yaml
```

## 5. 如何测试
```bash
python scripts/test_sr.py --config configs/default_sr.yaml --ckpt outputs/<exp>/checkpoints/best.pt --out outputs/test_eval
```

## 6. 如何查看输出结果
每次训练会新建 `outputs/<timestamp>/`，包含：
- `config.yaml`
- `train.log`
- `metrics.json`
- `metrics.csv`
- `checkpoints/latest.pt`, `checkpoints/best.pt`

## 7. 距离完整 O(2)-equivariant 论文版还差什么
- 严格群等变 encoder/decoder（当前仅接口占位）
- spectral basis / steerable kernel 模块
- 更精确的图像域 group action consistency + benchmark protocol


## 8. 离散 O(2) 近似说明（当前版本）
- 当前 `EquivariantEncoder` 是 lifting-based discrete group approximation：对每个离散群元素 `(k,r)` 先变换输入，再通过共享 CNN 编码。
- `GroupO2LIIFSR` 前向：`Enc_G -> GroupSampler -> GroupDecoder`。
- 这不是严格连续 O(2)-equivariant 网络；后续应替换为真实 group convolution / 连续群表示。
- 模型变体可通过 `model.variant` 切换：
  - `baseline_liif`
  - `baseline_liif_consistency`
  - `group_encoder_baseline_decoder`
  - `group_encoder_group_decoder`
  - `group_encoder_group_decoder_consistency`


## 9. Spectral parameterized group-aware SR (current stage)
- We insert spectral parameterization on group feature tensor `F_G` (scheme A): `Enc_G -> SpectralGroupBlocks -> Sampler -> Decoder`.
- Rotation dimension uses band-limited Fourier expansion (cos/sin real basis) with configurable `num_angular_modes=M`.
- Reflection dimension uses explicit even/odd decomposition: `even=(f0+f1)/2`, `odd=(f0-f1)/2`, then inverse recombination.
- This is **not** full continuous O(2) analytic convolution; it is a discrete group spectral approximation to reduce angular aliasing and representation inefficiency.

Run ablation scan:
```bash
python scripts/scan_spectral.py --config configs/default_sr.yaml --out outputs/spectral_scan.json
```



## 10. Continuous目标 vs 当前离散实现（重要）

本仓库当前实现应被视为对连续 O(2)-equivariant restoration 的**数值近似**，而不是理论终态：

### Continuous definitions (theoretical target)
- 真值图像：`x: Ω -> R^c`
- 观测：`y = S B x + n`
- 群作用：`[T_g x](u) = x(g^{-1}u), g in O(2)`
- latent：`z(u,g), (u,g) in Ω × O(2)`，并满足
  ` [T_h z](u,g) = z(h^{-1}u, h^{-1}g)`
- 输出：`x_hat(p) = D(p, A(z,p))`

### Discrete quadrature approximation (current code)
- `O(2)` 被离散化为 `G_K = {(k,r)}`，`k=0..K-1`, `r in {0,1}`。
- `EquivariantEncoder` 使用 lifting：`Enc_G(y) = {E_base(T_g y)}_{g in G_K}`。
- 这对应对群积分/群变量聚合的离散求和近似，不是连续群卷积解析形式。

### Finite angular truncation approximation (current code)
- `SpectralAngularMix` 只保留前 `M` 个角向模态（`num_angular_modes`），即 band-limited truncation。
- 使用实数 cos/sin basis 做投影与重构，并在每个 mode 上做可学习通道混合。
- reflection 通过 even/odd decomposition 进入：
  `even=(f0+f1)/2`, `odd=(f0-f1)/2`，分别谱变换后再重构。

### What is still missing for full continuous O(2)
- 连续群上的解析/可积核与严格等变算子（而非固定 K bins）。
- 从离散求和到可控连续积分近似误差界与采样策略。
- 与成像退化模型 (`S,B,n`) 更严格耦合的连续算子设计。
