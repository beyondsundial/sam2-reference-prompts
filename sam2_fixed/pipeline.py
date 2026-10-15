"""模型单次加载，多目标复用单张图的 image embedding，批量错误隔离。"""
import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from .prompts import read_rgb, validate_prompts, file_sha256
from .artifacts import save_segmentation_artifacts


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子替换避免进程退出留下半截 JSON。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_pipeline(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    targets, names = [], set()
    for item in config.get("targets", []):
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("target.name 必须为唯一的字母数字、横线或下划线")
        prompts_path = (path.parent / item["prompts"]).resolve()
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        validate_prompts(prompts)
        threshold = float(item.get("min_score", 0.0))
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("target.min_score 必须位于 [0,1]")
        targets.append({"name": name, "data": prompts, "min_score": threshold,
                        "prompts_path": str(prompts_path), "prompts_sha256": file_sha256(prompts_path)})
        names.add(name)
    if not targets:
        raise ValueError("pipeline 至少需要一个 target")
    shapes = {(target["data"]["image_size"]["height"], target["data"]["image_size"]["width"])
              for target in targets}
    if len(shapes) != 1:
        raise ValueError("同一流水线的目标提示必须使用相同图片尺寸")
    model = config["model"]
    return config, targets, str((path.parent / model["checkpoint"]).resolve())


def summarize_latency(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            **{f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}}


def run_batch(pipeline_path, images, output, warmup=1, runner=None):
    if not images or not 0 <= warmup <= 100:
        raise ValueError("需要输入图片；warmup 必须位于 [0,100]")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 一次运行一个输出目录，避免把上一批结果误认为本批结果。
    if any(output.iterdir()):
        raise ValueError("run 输出目录必须为空，请为每次运行指定新目录")
    config, targets, checkpoint = load_pipeline(pipeline_path)
    if runner is None:
        from .runner import FrozenSAM2
        model = config["model"]
        runner = FrozenSAM2(model["config"], checkpoint,
                            model.get("device", "auto"), model.get("precision", "float32"))
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": str(Path(pipeline_path).resolve()),
        "pipeline_sha256": file_sha256(pipeline_path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint) if Path(checkpoint).is_file() else None,
        "targets": [{key: value for key, value in target.items() if key != "data"} for target in targets],
        "warmup_requested": warmup, "warmup_completed": 0, "warmup_errors": [],
        "samples": [], "counts": {"accepted": 0, "rejected": 0, "error": 0},
        "timing_definition": "image read + model/point validation + artifact writes; excludes model load/warmup/JSON reports",
    }
    # 找到首张可用图预热；不把预热耗时混入正式分位数。
    if warmup:
        for image_path in images:
            try:
                rgb = read_rgb(image_path)
                for _ in range(warmup):
                    validate_prompts(targets[0]["data"], rgb.shape[:2])
                    runner.prepare(rgb)
                    for target in targets:
                        runner.segment_prepared(target["data"], target["min_score"])
                    summary["warmup_completed"] += 1
                break
            except (ValueError, OSError, RuntimeError) as error:
                summary["warmup_errors"].append({"image": str(image_path), "error": str(error)})
    latencies = []
    for index, image_path in enumerate(images):
        image_path = Path(image_path).resolve()
        sample = {"image": str(image_path), "targets": {}, "status": "accepted",
                  "mask_images": [], "mask_count": 0}
        start = time.perf_counter()
        folder = output / f"{index:04d}"
        folder.mkdir()
        try:
            rgb = read_rgb(image_path)
            sample["image_sha256"] = file_sha256(image_path)
            validate_prompts(targets[0]["data"], rgb.shape[:2])
            runner.prepare(rgb)
            for target in targets:
                try:
                    mask, metadata = runner.segment_prepared(target["data"], target["min_score"])
                    name = target["name"]
                    metadata.update(save_segmentation_artifacts(folder, name, rgb, target["data"], mask))
                    if "mask_path" in metadata:
                        sample["mask_images"].append({"target": name, "path": metadata["mask_path"]})
                        sample["mask_count"] = len(sample["mask_images"])
                    sample["targets"][name] = metadata
                    if mask is None and sample["status"] != "error":
                        sample["status"] = "rejected"
                except (ValueError, OSError, RuntimeError) as error:
                    sample["targets"][target["name"]] = {"status": "error", "error": str(error)}
                    sample["status"] = "error"
            sample["total_ms"] = (time.perf_counter() - start) * 1000
            # 失败与拒绝样本也保留耗时，不仅统计成功的快样本。
            latencies.append(sample["total_ms"])
        except (ValueError, OSError, RuntimeError) as error:
            sample.update({"status": "error", "error": str(error)})
        summary["counts"][sample["status"]] += 1
        summary["samples"].append(sample)
        write_json(folder / "result.json", sample)
        # 每张完成后写快照，便于中断后审计已完成结果。
        summary["latency"] = summarize_latency(latencies)
        write_json(output / "summary.json", summary)
    return summary
