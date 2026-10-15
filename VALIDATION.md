# 验证记录

日期：2026-10-15（Asia/Singapore）。此记录属于本次独立实现，不是历史公司项目的线上验收证据。

## 实际环境

| 项 | 实际值 |
|---|---|
| OS / 架构 | macOS 14.8.9 / arm64 |
| Python | 3.12.14 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 |
| NumPy / Pillow / OpenCV | 2.5.4 / 12.3.0 / 5.0.0 |
| SAM2 官方源码提交 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| 真模型验证 | 官方 SAM2.1 Hiera tiny，CPU，float32 |
| 权重 SHA256 | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |

依赖快照见 `requirements-tested-macos.txt`。已通过包安装、`pip check` 和真实模型推理。

## 自动化检查

启用真实权重执行 31 项测试，全部通过，无跳过；日志见 `evidence/tests-real-sam2.log`。覆盖：

- 环形与细小区域的内部取点、连通域、最小间距、正负重叠排除及点数预算。
- 空掩模、越界/冲突坐标、尺寸不符、配置摘要校验。
- RGB/RGBA 黑白掩模、透明背景、调色板与彩色原图误用检查。
- 候选拒绝、批量异常隔离、两个独立二值 PNG 的尺寸/像素值/输出数量。
- 图像编码复用、eval/无梯度、真实 SAM2 完整 state_dict 推理前后摘要一致。
- IoU/Dice、拒绝/错误计数、接受覆盖率及失败惩罚统计。

## 样本驱动检查

实际链路为参考掩模取点、生成固定坐标正负提示，再使用冻结 SAM2 推理。参考原图用于尺寸校验、来源追踪和预览；没有跨图片语义特征匹配。

检查记录见 `evidence/sample-driven-audit.json`。已验证提示可完全从当前样本重新生成，保存的 mask 与模型直接返回结果逐像素一致。旧演示脚本曾包含硬编码坐标及连通域后处理；该脚本已在清理中删除，不属于当前正式入口。

当前样本为 `samples/reference_masks/positive/circle`、`positive/square`、`negative/ellipse`。其完整掩模来自此前模型输出，作为演示参考，来源和摘要保存在 `provenance.json`。这些样本不是独立人工 GT，不据此报告泛化准确率。

## 当前结果

保留最近一次正式模块推理结果：

- `outputs/latest/summary.json`
- `outputs/latest/0000/result.json`
- `outputs/latest/0000/circle.mask.png`
- `outputs/latest/0000/square.mask.png`
- 两个目标各自的 `.overlay.png`

两个目标均有满足提示约束的候选，mask 为原图尺寸 1408×768 的 0/255 单通道 PNG。清理时仅迁移结果目录并修正 JSON 中的路径，未改变 mask 像素或分割代码。

旧的合成示例与多轮中间结果已删除。删除清单、空间统计及代码不变校验见 `evidence/cleanup.json`。

## 尚未验证

- 真实工业图像、独立标注测试集及原公司模块的一致性。
- large 权重、CUDA FP16/BF16、MPS 的实机效果与性能。
- 反光、遮挡、目标缺失、位置公差及换型边界。
- 毫米量测精度、批量尾延迟与 PLC/机器人集成。
- 远程 CI 运行。
