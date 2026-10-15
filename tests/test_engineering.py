import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam2_fixed.prompts import (build_prompts, select_points, validate_prompts,
                                select_candidate, read_mask, prompt_content_hash)
from sam2_fixed.pipeline import run_batch, load_pipeline, summarize_latency
from sam2_fixed.evaluation import binary_metrics, evaluate_run
from sam2_fixed.__main__ import main


def prompt(width=32, height=24):
    return {"schema_version": 1, "mode": "fixed_pixel_coordinates", "target": "pole",
            "image_size": {"width": width, "height": height},
            "point_coords": [[6, 6], [24, 16]], "point_labels": [1, 0]}


class SamplingTests(unittest.TestCase):
    def test_ring_point_stays_on_ring_not_center(self):
        yy, xx = np.indices((80, 80))
        radius = (xx - 40) ** 2 + (yy - 40) ** 2
        ring = (radius <= 30 ** 2) & (radius >= 18 ** 2)
        points = select_points(ring, points_per_component=4, min_spacing=15)
        self.assertEqual(len(points), 4)
        for x, y in points:
            self.assertTrue(ring[y, x])
            self.assertNotEqual((x, y), (40, 40))
        for index, a in enumerate(points):
            for b in points[index + 1:]:
                self.assertGreaterEqual(np.linalg.norm(np.subtract(a, b)), 15)

    def test_each_component_gets_a_point_and_noise_can_be_filtered(self):
        mask = np.zeros((50, 60), bool)
        mask[5:15, 7:17] = True
        mask[30:45, 40:55] = True
        mask[0, 0] = True
        points = select_points(mask, min_area=5)
        self.assertEqual(len(points), 2)
        self.assertTrue(all(mask[y, x] for x, y in points))

    def test_full_image_has_valid_boundary_distance(self):
        points = select_points(np.ones((9, 9), bool))
        self.assertEqual(points, [[4, 4]])

    def test_thin_mask_requires_explicit_lower_margin(self):
        mask = np.zeros((20, 20), bool)
        mask[5, 5:15] = True
        with self.assertRaises(ValueError):
            select_points(mask)
        self.assertTrue(select_points(mask, boundary_margin=1))

    def test_invalid_sampling_parameters_fail(self):
        for kwargs in [{"min_spacing": 0}, {"boundary_margin": float("nan")},
                       {"points_per_component": 0}, {"min_area": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_points(np.ones((10, 10), bool), **kwargs)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(self.root / "ref.png")
        self.positive = np.zeros((24, 32), bool)
        self.positive[2:14, 2:14] = True
        self.negative = np.zeros((24, 32), bool)
        self.negative[8:22, 8:30] = True
        self.save_mask("positive.png", self.positive)
        self.save_mask("negative.png", self.negative)
        self.manifest = {"target": "pole", "positive": [{"image": "ref.png", "mask": "positive.png"}],
                         "negative": [{"image": "ref.png", "mask": "negative.png"}]}
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def save_mask(self, name, mask):
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.root / name)

    def write_manifest(self):
        path = self.root / "references.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return path

    def test_negative_points_never_hit_any_positive_reference_region(self):
        data, _ = build_prompts(self.write_manifest())
        self.assertGreater(data["negative_overlap_pixels_removed"], 0)
        for (x, y), label in zip(data["point_coords"], data["point_labels"]):
            if label == 0:
                self.assertTrue(self.negative[y, x])
                self.assertFalse(self.positive[y, x])

    def test_positive_points_come_from_stable_intersection(self):
        shifted = np.roll(self.positive, 3, axis=1)
        self.save_mask("positive2.png", shifted)
        self.manifest["positive"].append({"image": "ref.png", "mask": "positive2.png"})
        data, _ = build_prompts(self.write_manifest())
        for (x, y), label in zip(data["point_coords"], data["point_labels"]):
            if label == 1:
                self.assertTrue((shifted & self.positive)[y, x])

    def test_no_common_positive_region_fails(self):
        other = np.zeros_like(self.positive)
        other[16:22, 24:30] = True
        self.save_mask("other.png", other)
        self.manifest["positive"].append({"image": "ref.png", "mask": "other.png"})
        with self.assertRaisesRegex(ValueError, "共同区域"):
            build_prompts(self.write_manifest())

    def test_fully_conflicting_negative_fails(self):
        self.save_mask("negative.png", self.positive)
        with self.assertRaisesRegex(ValueError, "完全覆盖"):
            build_prompts(self.write_manifest())

    def test_revision_is_stable_and_detects_edits(self):
        first, _ = build_prompts(self.write_manifest())
        second, _ = build_prompts(self.write_manifest())
        self.assertEqual(first["prompt_id"], second["prompt_id"])
        changed = copy.deepcopy(first)
        changed["point_coords"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "prompt_id"):
            validate_prompts(changed)

    def test_mask_image_size_mismatch_fails(self):
        self.save_mask("positive.png", np.ones((10, 10), bool))
        with self.assertRaisesRegex(ValueError, "尺寸"):
            build_prompts(self.write_manifest())

    def test_empty_and_colored_masks_fail(self):
        self.save_mask("empty.png", np.zeros((24, 32), bool))
        with self.assertRaises(ValueError):
            read_mask(self.root / "empty.png")
        colored = np.zeros((24, 32, 3), np.uint8)
        colored[5:15, 5:15] = [255, 0, 0]
        Image.fromarray(colored).save(self.root / "colored.png")
        with self.assertRaisesRegex(ValueError, "彩色"):
            read_mask(self.root / "colored.png")

    def test_rgb_and_rgba_black_white_masks_match_grayscale(self):
        for mode in ["RGB", "RGBA"]:
            with Image.open(self.root / "positive.png") as image:
                image.convert(mode).save(self.root / f"{mode}.png")
            np.testing.assert_array_equal(read_mask(self.root / f"{mode}.png"), self.positive)

    def test_transparent_white_pixels_are_background(self):
        rgba = np.full((24, 32, 4), 255, np.uint8)
        rgba[:, :, 3] = 0
        rgba[5:10, 5:10, 3] = 255
        Image.fromarray(rgba).save(self.root / "transparent.png")
        expected = np.zeros((24, 32), bool)
        expected[5:10, 5:10] = True
        np.testing.assert_array_equal(read_mask(self.root / "transparent.png"), expected)

    def test_palette_indices_do_not_invert_white_foreground(self):
        indices = np.ones((24, 32), np.uint8)
        indices[self.positive] = 0  # 索引0=白，索引1=黑。
        image = Image.fromarray(indices).convert("P")
        image.putpalette([255, 255, 255, 0, 0, 0] + [0] * (256 * 3 - 6))
        image.save(self.root / "palette.png")
        np.testing.assert_array_equal(read_mask(self.root / "palette.png"), self.positive)

    def test_empty_mask_is_allowed_only_when_explicit(self):
        self.save_mask("empty.png", np.zeros((24, 32), bool))
        self.assertFalse(read_mask(self.root / "empty.png", allow_empty=True).any())

    def test_point_budget_fails_instead_of_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "上限"):
            build_prompts(self.write_manifest(), max_points=1)

    def test_build_and_preview_cli_roundtrip(self):
        output = self.root / "prompts.json"
        self.assertEqual(main(["build", "--manifest", str(self.write_manifest()), "--out", str(output)]), 0)
        self.assertTrue(output.with_suffix(".preview.png").is_file())
        self.assertEqual(main(["preview", "--prompts", str(output), "--image", str(self.root / "ref.png"),
                               "--out", str(self.root / "query.preview.png")]), 0)


class ValidationTests(unittest.TestCase):
    def test_wrong_size_is_never_rescaled(self):
        with self.assertRaisesRegex(ValueError, "尺寸一致"):
            validate_prompts(prompt(), (48, 64))

    def test_bad_coordinates_and_labels_fail(self):
        for coords, labels in [([[32, 6]], [1]), ([[float("nan"), 6]], [1]),
                               ([[1.5, 6]], [1]), ([[6, 6]], [2]),
                               ([[6, 6]], [0]), ([[6, 6], [6, 6]], [1, 0])]:
            data = prompt()
            data.update(point_coords=coords, point_labels=labels)
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                validate_prompts(data)

    def test_higher_scored_invalid_candidate_is_rejected(self):
        masks = np.zeros((3, 24, 32), bool)
        masks[0, 6, 6] = True
        masks[1] = True
        index, results = select_candidate(masks, [0.6, 0.99, 0.9], prompt())
        self.assertEqual(index, 0)
        self.assertFalse(results[1]["negative_points_satisfied"])
        self.assertFalse(results[2]["positive_points_satisfied"])

    def test_no_valid_candidate_does_not_force_output(self):
        index, _ = select_candidate(np.zeros((1, 24, 32), bool), [0.99], prompt())
        self.assertIsNone(index)

    def test_nonfinite_score_and_quality_threshold_reject(self):
        masks = np.zeros((1, 24, 32), bool)
        masks[0, 6, 6] = True
        self.assertIsNone(select_candidate(masks, [float("nan")], prompt())[0])
        self.assertIsNone(select_candidate(masks, [0.6], prompt(), min_score=0.7)[0])


class FakePredictor:
    """仅用于接口契约测试；不模拟 SAM2 的分割准确率。使用真实 PyTorch。"""
    def __init__(self):
        import torch
        self.model = torch.nn.Linear(2, 2)
        self.encode_calls = 0

    def set_image(self, rgb):
        import torch
        assert torch.is_inference_mode_enabled()
        assert not self.model.training
        self.model(torch.zeros(1, 2))
        self.hw = rgb.shape[:2]
        self.reject = rgb[0, 0, 0] == 255
        self.encode_calls += 1

    def predict(self, point_coords, point_labels, **kwargs):
        import torch
        assert torch.is_inference_mode_enabled()
        masks = np.zeros((1, *self.hw), bool)
        if not self.reject:
            for (x, y), label in zip(point_coords.astype(int), point_labels):
                masks[0, y, x] = bool(label)
        return masks, np.array([0.9]), None


class FrozenAndPipelineTests(unittest.TestCase):
    def make_runner(self):
        from sam2_fixed.runner import FrozenSAM2
        return FrozenSAM2.from_predictor(FakePredictor())

    def test_eval_no_grad_and_parameters_unchanged(self):
        import torch
        runner = self.make_runner()
        before = [p.detach().clone() for p in runner.predictor.model.parameters()]
        mask, meta = runner.segment(np.zeros((24, 32, 3), np.uint8), prompt())
        self.assertIsNotNone(mask)
        self.assertEqual(meta["status"], "accepted")
        self.assertFalse(runner.predictor.model.training)
        for original, parameter in zip(before, runner.predictor.model.parameters()):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(original, parameter))

    def test_prepare_is_required_and_embedding_can_be_reused(self):
        runner = self.make_runner()
        with self.assertRaises(ValueError):
            runner.segment_prepared(prompt())
        runner.prepare(np.zeros((24, 32, 3), np.uint8))
        runner.segment_prepared(prompt())
        runner.segment_prepared(prompt())
        self.assertEqual(runner.predictor.encode_calls, 1)

    def test_failed_prepare_clears_previous_image_state(self):
        runner = self.make_runner()
        runner.prepare(np.zeros((24, 32, 3), np.uint8))
        with self.assertRaises(ValueError):
            runner.prepare(np.zeros((24, 32), np.uint8))
        with self.assertRaises(ValueError):
            runner.segment_prepared(prompt())

    def test_batch_isolates_bad_image_and_reports_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = np.zeros((24, 32, 3), np.uint8)
            Image.fromarray(rgb).save(root / "good.png")
            rgb[0, 0] = 255
            Image.fromarray(rgb).save(root / "reject.png")
            (root / "bad.png").write_text("broken", encoding="utf-8")
            (root / "prompts.json").write_text(json.dumps(prompt()), encoding="utf-8")
            pipeline = {"model": {"config": "unused", "checkpoint": "missing.pt"},
                        "targets": [{"name": "pole", "prompts": "prompts.json"},
                                    {"name": "weld", "prompts": "prompts.json"}]}
            pipeline_path = root / "pipeline.json"
            pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
            runner = self.make_runner()
            summary = run_batch(pipeline_path, [root / "good.png", root / "bad.png", root / "reject.png"],
                                root / "run", warmup=1, runner=runner)
            self.assertEqual(summary["counts"], {"accepted": 1, "rejected": 1, "error": 1})
            self.assertEqual(summary["warmup_completed"], 1)
            self.assertEqual(runner.predictor.encode_calls, 3)  # 预热+两张可读图；每张2目标只编码1次。
            self.assertTrue((root / "run/0000/pole.mask.png").is_file())
            self.assertTrue((root / "run/0000/weld.mask.png").is_file())
            self.assertEqual(summary["samples"][0]["mask_count"], 2)
            self.assertEqual({item["target"] for item in summary["samples"][0]["mask_images"]}, {"pole", "weld"})
            for name in ["pole", "weld"]:
                with Image.open(root / f"run/0000/{name}.mask.png") as image:
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (32, 24))
                    self.assertEqual(set(np.unique(np.asarray(image))), {0, 255})
            self.assertFalse((root / "run/0002/pole.mask.png").exists())
            self.assertEqual(summary["samples"][1]["mask_count"], 0)
            self.assertEqual(summary["samples"][2]["mask_count"], 0)
            truth = np.zeros((24, 32), np.uint8)
            truth[6, 6] = 255
            Image.fromarray(truth).save(root / "gt.png")
            manifest = {"samples": [{"image": name, "mask": "gt.png", "target": "pole"}
                                    for name in ["good.png", "bad.png", "reject.png"]]}
            (root / "eval.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = evaluate_run(root / "run/summary.json", root / "eval.json")
            self.assertEqual(report["aggregate"]["accepted_mean_iou"], 1.0)
            self.assertAlmostEqual(report["aggregate"]["coverage"], 1 / 3)
            self.assertAlmostEqual(report["aggregate"]["failure_penalized_mean_iou"], 1 / 3)
            with self.assertRaisesRegex(ValueError, "为空"):
                run_batch(pipeline_path, [root / "good.png"], root / "run", runner=runner)


class MetricsTests(unittest.TestCase):
    def test_iou_dice_known_overlap(self):
        result = binary_metrics([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["dice"], 0.5)

    def test_empty_pair_and_shape_mismatch(self):
        self.assertEqual(binary_metrics(np.zeros((3, 3)), np.zeros((3, 3)))["iou"], 1)
        with self.assertRaises(ValueError):
            binary_metrics(np.zeros((3, 3)), np.zeros((2, 2)))

    def test_latency_count_and_percentiles(self):
        result = summarize_latency([1, 2, 3, 4, 5])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["p50_ms"], 3)
        self.assertGreaterEqual(result["p99_ms"], result["p95_ms"])


@unittest.skipUnless(os.environ.get("SAM2_TEST_CHECKPOINT"), "未提供真实 SAM2 权重；不是已验证的模型效果测试")
class RealSAM2SmokeTest(unittest.TestCase):
    def test_actual_model_freeze_and_prediction(self):
        import torch
        from sam2_fixed.runner import FrozenSAM2
        runner = FrozenSAM2(os.environ.get("SAM2_TEST_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml"),
                            os.environ["SAM2_TEST_CHECKPOINT"], device="cpu")
        image = np.zeros((24, 32, 3), np.uint8)
        def state_digest():
            digest = hashlib.sha256()
            for name, value in sorted(runner.predictor.model.state_dict().items()):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        before = state_digest()
        _, metadata = runner.segment(image, prompt())
        self.assertIn(metadata["status"], ("accepted", "rejected"))
        self.assertEqual(before, state_digest())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in runner.predictor.model.parameters()))


if __name__ == "__main__":
    unittest.main()
