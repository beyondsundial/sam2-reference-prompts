"""统一输出独立二值 mask 与可视化叠加图。"""
from pathlib import Path

import numpy as np
from PIL import Image

from .prompts import draw_overlay


def save_segmentation_artifacts(folder, stem, rgb, prompts, mask):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / f"{stem}.mask.png"
    artifacts = {}
    if mask is not None:
        binary = np.asarray(mask, dtype=bool)
        if binary.shape != rgb.shape[:2]:
            raise ValueError("输出 mask 必须与原图尺寸一致")
        # 独立单通道 PNG：目标=255，背景=0，不含原图、颜色或提示点。
        Image.fromarray(binary.astype(np.uint8) * 255).save(mask_path)
        artifacts["mask_path"] = str(mask_path)
    else:
        # 拒绝时清理可能存在的旧结果，不能用全黑图冒充有效分割。
        mask_path.unlink(missing_ok=True)
    overlay_path = folder / f"{stem}.overlay.png"
    draw_overlay(rgb, prompts, mask).save(overlay_path)
    artifacts["overlay_path"] = str(overlay_path)
    return artifacts
