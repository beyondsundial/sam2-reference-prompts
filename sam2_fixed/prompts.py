"""从同一坐标系内的参考掩模提取点；不接受待检图，不做特征匹配。"""
import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def read_mask(path, allow_empty=False):
    """读取黑白标注；RGB/RGBA 仅接受灰度内容，透明像素视为背景。"""
    with Image.open(path) as image:
        if image.mode in ("1", "L", "I", "I;16"):
            mask = np.asarray(image) != 0
        elif image.mode in ("RGB", "RGBA", "LA", "P"):
            rgba = np.asarray(image.convert("RGBA"))
            rgb, visible = rgba[:, :, :3], rgba[:, :, 3] != 0
            gray_pixels = (rgb[:, :, 0] == rgb[:, :, 1]) & (rgb[:, :, 1] == rgb[:, :, 2])
            if (visible & ~gray_pixels).any():
                raise ValueError(f"掩模必须是黑白/灰度标注，不能直接使用彩色原图或彩色叠加图: {path}")
            # P 模式按实际调色板颜色解释，不能假设索引0必定是黑色。
            mask = (rgb[:, :, 0] != 0) & visible
        else:
            raise ValueError(f"不支持的掩模格式 {image.mode}: {path}")
    if not allow_empty and not mask.any():
        raise ValueError(f"掩模为空: {path}")
    return mask


def select_points(mask, points_per_component=1, min_area=1,
                  boundary_margin=2.0, min_spacing=12.0):
    """每个连通域先选距边界最远的点，再按间距选内部点。返回 (x,y)。"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("取点掩模必须为非空二维数组")
    if points_per_component < 1 or min_area < 1:
        raise ValueError("points_per_component 和 min_area 必须为正整数")
    if not math.isfinite(boundary_margin) or boundary_margin < 0:
        raise ValueError("boundary_margin 必须为有限非负数")
    if not math.isfinite(min_spacing) or min_spacing <= 0:
        raise ValueError("min_spacing 必须为有限正数")
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    points = []
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] < min_area:
            continue
        left, top, width, height, _ = stats[component_id]
        region = components[top:top + height, left:left + width] == component_id
        # 补零使接触图像边缘的区域也有明确边界。
        distance = cv2.distanceTransform(
            np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )[1:-1, 1:-1]
        eligible = region & (distance >= boundary_margin)
        if not eligible.any():
            raise ValueError(
                f"连通域 {component_id} 无满足边界距离的点；减小 boundary_margin 或检查掩模"
            )
        yy, xx = np.indices(region.shape)
        for _ in range(points_per_component):
            if not eligible.any():
                break
            ranked = np.where(eligible, distance, -1)
            y, x = np.unravel_index(np.argmax(ranked), ranked.shape)
            points.append([int(x + left), int(y + top)])
            eligible &= (xx - x) ** 2 + (yy - y) ** 2 >= min_spacing ** 2
    if not points:
        raise ValueError("没有可用提示点；检查 min_area 与掩模")
    return points


def validate_prompts(data, image_hw=None):
    if data.get("mode") != "fixed_pixel_coordinates":
        raise ValueError("配置必须声明 fixed_pixel_coordinates 模式")
    if data.get("schema_version") != 1:
        raise ValueError("不支持的提示配置版本")
    if "prompt_id" in data and data["prompt_id"] != prompt_content_hash(data):
        raise ValueError("提示配置内容与 prompt_id 不一致；请重新 build")
    size = data.get("image_size", {})
    width, height = size.get("width"), size.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("无效的参考图片尺寸")
    if image_hw is not None and tuple(image_hw) != (height, width):
        raise ValueError(
            f"固定坐标模式要求尺寸一致：参考 {width}x{height}，待检 {image_hw[1]}x{image_hw[0]}"
        )
    coords = np.asarray(data.get("point_coords", []), dtype=np.float32)
    labels = np.asarray(data.get("point_labels", []))
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("point_coords 必须为非空 Nx2 数组")
    if labels.shape != (len(coords),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("point_labels 必须与坐标等长且仅含0/1")
    if not (labels == 1).any():
        raise ValueError("至少需要一个正点")
    if not np.isfinite(coords).all() or not np.equal(coords, np.round(coords)).all():
        raise ValueError("固定提示坐标必须是有限整数像素坐标")
    if (coords < 0).any() or (coords[:, 0] >= width).any() or (coords[:, 1] >= height).any():
        raise ValueError("提示点越界")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("存在重复或正负冲突的提示点")
    return coords, labels.astype(np.int32)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_content_hash(data):
    content = {key: value for key, value in data.items() if key not in ("prompt_id", "created_at_utc")}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_prompts(manifest_path, max_points=64, **sampling):
    """正掩模取交集获得稳定位置；负掩模取并集并排除所有正参考区域。"""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    positive_masks, negative_masks, sources = [], [], []
    expected_hw, preview_image = None, None
    for kind, destination in [("positive", positive_masks), ("negative", negative_masks)]:
        for sample in manifest.get(kind, []):
            image_path = (manifest_path.parent / sample["image"]).resolve()
            mask_path = (manifest_path.parent / sample["mask"]).resolve()
            image, mask = read_rgb(image_path), read_mask(mask_path)
            if image.shape[:2] != mask.shape:
                raise ValueError(f"参考原图与 mask 尺寸不一致: {mask_path}")
            if expected_hw is None:
                expected_hw, preview_image = mask.shape, image
            if mask.shape != expected_hw:
                raise ValueError("全部参考样本必须使用相同尺寸和坐标系，禁止自动缩放")
            destination.append(mask)
            sources.append({"kind": kind, "image": str(image_path), "mask": str(mask_path),
                            "image_sha256": file_sha256(image_path),
                            "mask_sha256": file_sha256(mask_path)})
    if not positive_masks:
        raise ValueError("manifest 至少需要一个 positive 样本")
    stable_positive = np.logical_and.reduce(positive_masks)
    if not stable_positive.any():
        raise ValueError("正参考掩模没有共同区域；请检查工件位置或按工位/型号分配置")
    all_positive = np.logical_or.reduce(positive_masks)
    all_negative = (np.logical_or.reduce(negative_masks) if negative_masks
                    else np.zeros(expected_hw, dtype=bool))
    # 固定坐标下，把任何参考中属于目标的位置当负点都可能误抑制目标。
    safe_negative = all_negative & ~all_positive
    if negative_masks and not safe_negative.any():
        raise ValueError("负掩模被正参考区域完全覆盖；无法生成安全负点")
    positive_points = select_points(stable_positive, **sampling)
    negative_points = select_points(safe_negative, **sampling) if safe_negative.any() else []
    if max_points < 1 or len(positive_points) + len(negative_points) > max_points:
        raise ValueError(f"提示数量超过上限 {max_points}；检查噪声连通域或调整 min_area")
    height, width = expected_hw
    data = {
        "schema_version": 1,
        "mode": "fixed_pixel_coordinates",
        "target": manifest.get("target", "target"),
        "image_size": {"width": width, "height": height},
        "point_coords": positive_points + negative_points,
        "point_labels": [1] * len(positive_points) + [0] * len(negative_points),
        "sampling": sampling,
        "sources": sources,
        "aggregation": "positive_intersection; negative_union_excluding_positive_union",
        "negative_overlap_pixels_removed": int((all_negative & all_positive).sum()),
        "max_points": max_points,
    }
    # 相同数据与参数有相同版本号；时间戳不参与摘要。
    data["prompt_id"] = prompt_content_hash(data)
    data["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    validate_prompts(data)
    return data, preview_image


def draw_overlay(rgb, data, mask=None):
    canvas = rgb.copy()
    if mask is not None:
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array([30, 180, 255])).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index, ((x, y), label) in enumerate(zip(data["point_coords"], data["point_labels"])):
        color = "lime" if label == 1 else "red"
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="black")
        draw.text((x + 6, y - 5), f"{index}:{label}", fill=color, stroke_width=1, stroke_fill="black")
    return image


def select_candidate(masks, scores, data, min_score=0.0):
    """只在满足所有正负点的候选中按模型预测质量选优；不修改边界。"""
    coords, labels = validate_prompts(data)
    masks, scores = np.asarray(masks, dtype=bool), np.asarray(scores)
    expected = (data["image_size"]["height"], data["image_size"]["width"])
    if masks.ndim != 3 or masks.shape[1:] != expected or scores.shape != (len(masks),):
        raise ValueError("SAM2 返回的掩模/分数形状不正确")
    xy = coords.astype(np.int64)
    results, valid = [], []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        point_values = mask[xy[:, 1], xy[:, 0]]
        positive_ok = bool(point_values[labels == 1].all())
        negative_ok = bool((~point_values[labels == 0]).all())
        finite_score = bool(np.isfinite(score))
        accepted = positive_ok and negative_ok and finite_score and float(score) >= min_score
        results.append({"index": index, "predicted_quality": float(score) if finite_score else None,
                        "positive_points_satisfied": positive_ok,
                        "negative_points_satisfied": negative_ok, "accepted": accepted})
        if accepted:
            valid.append(index)
    best = max(valid, key=lambda index: scores[index]) if valid else None
    return best, results
