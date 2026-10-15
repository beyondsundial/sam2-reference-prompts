"""离线评估；拒绝与错误单列，避免只汇报成功样本的 IoU。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .prompts import read_mask


def binary_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or truth.ndim != 2:
        raise ValueError("预测与 GT 必须为尺寸相同的二维掩模")
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    total = int(prediction.sum() + truth.sum())
    return {"iou": intersection / union if union else 1.0,
            "dice": 2 * intersection / total if total else 1.0}


def evaluate_run(run_path, manifest_path):
    run_path, manifest_path = Path(run_path).resolve(), Path(manifest_path).resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_image = {}
    for sample in run["samples"]:
        key = str(Path(sample["image"]).resolve())
        if key in by_image:
            raise ValueError("同一图片在运行结果中重复，无法唯一匹配 GT")
        by_image[key] = sample
    rows = []
    seen = set()
    for item in manifest["samples"]:
        image_path = str((manifest_path.parent / item["image"]).resolve())
        target = item["target"]
        if (image_path, target) in seen:
            raise ValueError("评估清单存在重复图片/目标")
        seen.add((image_path, target))
        gt_path = (manifest_path.parent / item["mask"]).resolve()
        truth = read_mask(gt_path, allow_empty=True)
        sample = by_image.get(image_path)
        if sample is None:
            raise ValueError(f"评估图片未出现在运行结果中: {image_path}")
        result = sample.get("targets", {}).get(target)
        if result is None and sample["status"] != "error":
            raise ValueError(f"运行结果未包含目标 {target}")
        row = {"image": image_path, "target": target,
               "status": result["status"] if result else "error",
               "iou": None, "dice": None}
        if row["status"] == "accepted":
            with Image.open(result["mask_path"]) as image:
                prediction = np.asarray(image) != 0
            row.update(binary_metrics(prediction, truth))
        rows.append(row)
    if not rows:
        raise ValueError("评估清单为空")
    groups = {name: [row for row in rows if row["target"] == name]
              for name in sorted({row["target"] for row in rows})}
    def aggregate(group):
        accepted = [row for row in group if row["status"] == "accepted"]
        return {"count": len(group), "accepted": len(accepted),
                "rejected": sum(row["status"] == "rejected" for row in group),
                "errors": sum(row["status"] == "error" for row in group),
                "coverage": len(accepted) / len(group),
                "accepted_mean_iou": float(np.mean([row["iou"] for row in accepted])) if accepted else None,
                "accepted_mean_dice": float(np.mean([row["dice"] for row in accepted])) if accepted else None,
                "failure_penalized_mean_iou": sum(row["iou"] or 0 for row in group) / len(group)}
    return {"run": str(run_path), "manifest": str(manifest_path),
            "aggregate": aggregate(rows),
            "by_target": {name: aggregate(group) for name, group in groups.items()},
            "samples": rows,
            "metric_note": "accepted_mean_* 仅统计接受样本；failure_penalized_mean_iou 将拒绝和错误按0计，非标准语义分割mIoU"}
