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
把图像放到 `data/train_hr`, `data/val_hr`, `data/test_hr`。

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
