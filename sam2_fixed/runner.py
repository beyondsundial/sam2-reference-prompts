"""仅推理：eval + requires_grad_(False) + inference_mode；不构造优化器。"""
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from .prompts import validate_prompts, select_candidate


class FrozenSAM2:
    def __init__(self, config, checkpoint, device="auto", precision="float32"):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError("SAM2/PyTorch 未安装；请按 README 安装官方 SAM2") from error
        if not Path(checkpoint).is_file():
            raise ValueError(f"权重不存在: {checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS 不可用")
        if device not in ("cuda", "cpu", "mps"):
            raise ValueError("device 必须为 auto/cuda/cpu/mps")
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("不支持的精度")
        if device != "cuda" and precision != "float32":
            raise ValueError("混合精度仅在 CUDA 下启用；其他设备请选择 float32")
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 GPU 不支持 bfloat16，请使用 float16 或 float32")
        self.torch, self.device, self.precision = torch, device, precision
        model = build_sam2(config, str(Path(checkpoint).resolve()), device=device, mode="eval")
        model.eval()
        model.requires_grad_(False)
        self.predictor = SAM2ImagePredictor(model)

    @classmethod
    def from_predictor(cls, predictor):
        """接入已有 CPU predictor，也用于不下载权重的契约集成测试。"""
        import torch
        instance = cls.__new__(cls)
        instance.torch, instance.device, instance.precision = torch, "cpu", "float32"
        predictor.model.eval()
        predictor.model.requires_grad_(False)
        instance.predictor = predictor
        return instance

    def _autocast(self):
        return (self.torch.autocast("cuda", dtype=getattr(self.torch, self.precision))
                if self.precision != "float32" else nullcontext())

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def prepare(self, rgb):
        """每张图调用一次；多目标复用图像编码结果。实例不支持并发请求。"""
        self._image_hw = None
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("输入必须为 HWC uint8 RGB 图片")
        self._sync()
        start = time.perf_counter()
        with self.torch.inference_mode(), self._autocast():
            self.predictor.set_image(rgb)
        self._sync()
        self._encode_ms = (time.perf_counter() - start) * 1000
        self._image_hw = rgb.shape[:2]

    def segment(self, rgb, data, min_score=0.0):
        validate_prompts(data, rgb.shape[:2])
        self.prepare(rgb)
        return self.segment_prepared(data, min_score)

    def segment_prepared(self, data, min_score=0.0):
        if getattr(self, "_image_hw", None) is None:
            raise ValueError("请先 prepare 当前图片")
        coords, labels = validate_prompts(data, self._image_hw)
        if not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score 必须在 [0,1] 内")
        self._sync()
        start = time.perf_counter()
        with self.torch.inference_mode(), self._autocast():
            masks, scores, _ = self.predictor.predict(
                point_coords=coords, point_labels=labels,
                multimask_output=True, normalize_coords=True,
            )
        self._sync()
        best, candidates = select_candidate(masks, scores, data, min_score)
        metadata = {
            "status": "accepted" if best is not None else "rejected",
            "selected_index": best, "candidates": candidates,
            "decode_and_selection_ms": (time.perf_counter() - start) * 1000,
            "shared_image_encode_ms": self._encode_ms,
            "device": self.device, "precision": self.precision,
            "mode": "fixed_pixel_coordinates", "weights_frozen": True,
            "prompt_id": data.get("prompt_id"),
        }
        return (np.asarray(masks[best], dtype=bool) if best is not None else None), metadata
