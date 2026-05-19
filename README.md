# Continuous-Rotation: O2LIIFSR / GroupO2LIIFSR Research Prototype

## 1) 结论先说：是否已经实现“连续旋转等变”？
**没有。当前版本还不是严格的 continuous O(2)-equivariant network。**

当前实现是一个**离散近似原型**：
- 用离散群 `G_K={(k,r)}`（`k=0..K-1`, `r∈{0,1}`）近似 `O(2)`；
- `EquivariantEncoder` 采用 lifting（对每个离散群元素变换输入后共享 backbone 编码）；
- `SpectralAngularMix` 在离散 rotation bins 上做有限阶 Fourier 模态截断（`num_angular_modes=M`）；
- reflection 通过 even/odd decomposition 处理。

所以它是：
- ✅ **discrete quadrature approximation + finite angular truncation**
- ❌ **不是最终连续群解析等变算子**。

---

## 2) 当前实现了什么
- Arbitrary-scale implicit SR 主链路：`Enc -> Samp -> Dec`
- Group-aware 版本：`Enc_G -> SpectralGroupBlocks -> GroupSampler -> GroupDecoder`
- 训练损失：`L_rec + λ_rot L_rot + λ_ref L_ref`
- 指标：`PSNR / SSIM / Rot-EE / Ref-EE`
- 自动化：数据准备、训练、测试、checkpoint、指标导出

---

## 3) 如何运行代码

### 3.1 准备数据目录
```bash
python scripts/prepare_data.py --root ./data
```
- 会创建：`data/train_hr`, `data/val_hr`, `data/test_hr`
- 会尝试下载公开 DIV2K（若网络不可达会给出 `download_report`）

若你只想建目录不下载：
```bash
python scripts/prepare_data.py --root ./data --no-download
```

### 3.2 训练
```bash
python scripts/train_sr.py --config configs/default_sr.yaml
```
默认配置会在 `outputs/<timestamp>/` 生成：
- `config.yaml`
- `train.log`
- `metrics.json`
- `metrics.csv`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`

### 3.3 测试
```bash
python scripts/test_sr.py \
  --config configs/default_sr.yaml \
  --ckpt outputs/<exp>/checkpoints/best.pt \
  --out outputs/test_eval
```

### 3.4 谱参数扫描（消融）
```bash
python scripts/scan_spectral.py --config configs/default_sr.yaml --out outputs/spectral_scan.json
```
会扫：
- `K in {4,8,12,16}`
- `use_reflection in {False,True}`
- `M in [1..floor(K/2)]`
并输出 JSON/CSV。

---

## 4) 配置切换（重点）
在 `configs/default_sr.yaml` 里可改：
- `model.variant`：
  - `baseline_liif`
  - `baseline_liif_consistency`
  - `group_encoder_baseline_decoder`
  - `group_encoder_group_decoder`
  - `group_encoder_group_decoder_consistency`
- `model.num_rotations` (`K`)
- `model.use_reflection`
- `model.num_angular_modes` (`M`)
- `model.num_spectral_blocks`
- `model.spectral_residual`
- `model.enable_spectral`

---

## 5) 与最终“连续 O(2)-equivariant restoration”还差什么
- 连续群上的严格等变卷积/算子（非固定 `K` bins）
- 连续群积分与离散求和误差控制（可证明近似误差）
- 与成像退化模型 `y=SBx+n` 深度耦合的连续算子实现
- 更完整的连续角度评估协议与理论验证

---

## 6) 一句话定位
这个仓库现在是：
> **可运行的、研究导向的、离散 O(2) 近似 + 谱截断的 implicit SR 原型**，
> 不是最终连续 O(2)-equivariant 论文模型。
