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
