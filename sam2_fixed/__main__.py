import argparse
import json
from pathlib import Path
import sys

from .prompts import build_prompts, draw_overlay, read_rgb, validate_prompts
from .artifacts import save_segmentation_artifacts


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="参考 mask 固定提示点 → 冻结 SAM2 分割")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从参考样本生成固定提示 JSON 与预览")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--points-per-component", type=int, default=1)
    build.add_argument("--min-area", type=int, default=1)
    build.add_argument("--boundary-margin", type=float, default=2.0)
    build.add_argument("--min-spacing", type=float, default=12.0)
    build.add_argument("--max-points", type=int, default=64)
    preview = commands.add_parser("preview", help="在待检图上检查固定点；不加载模型")
    preview.add_argument("--prompts", type=Path, required=True)
    preview.add_argument("--image", type=Path, required=True)
    preview.add_argument("--out", type=Path, required=True)
    predict = commands.add_parser("predict", help="加载一次模型，逐张分割")
    predict.add_argument("--prompts", type=Path, required=True)
    predict.add_argument("--image", type=Path, nargs="+", required=True)
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    predict.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    predict.add_argument("--min-score", type=float, default=0.0)
    predict.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="多目标批量流水线；逐图隔离错误并汇总")
    run.add_argument("--pipeline", type=Path, required=True)
    run.add_argument("--image", type=Path, nargs="+", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--warmup", type=int, default=1)
    evaluate = commands.add_parser("evaluate", help="用独立人工标注评估已保存的推理结果")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .pipeline import run_batch
            summary = run_batch(args.pipeline, args.image, args.out, warmup=args.warmup)
            print(json.dumps(summary["counts"], ensure_ascii=False))
            for sample in summary["samples"]:
                print(f"{Path(sample['image']).name}: 已输出 {sample['mask_count']} 张独立 mask 图")
                for item in sample["mask_images"]:
                    print(f"  {item['target']}: {item['path']}")
            return 0 if summary["counts"]["accepted"] == len(summary["samples"]) else 2
        elif args.command == "evaluate":
            from .evaluation import evaluate_run
            report = evaluate_run(args.run, args.manifest)
            save_json(args.out, report)
            print(json.dumps(report["aggregate"], ensure_ascii=False))
        elif args.command == "build":
            data, image = build_prompts(
                args.manifest, points_per_component=args.points_per_component,
                min_area=args.min_area, boundary_margin=args.boundary_margin,
                min_spacing=args.min_spacing,
                max_points=args.max_points,
            )
            save_json(args.out, data)
            draw_overlay(image, data).save(args.out.with_suffix(".preview.png"))
            print(f"已生成 {len(data['point_coords'])} 个提示点: {args.out}")
        else:
            data = json.loads(args.prompts.read_text(encoding="utf-8"))
            validate_prompts(data)
            if args.command == "preview":
                image = read_rgb(args.image)
                validate_prompts(data, image.shape[:2])
                args.out.parent.mkdir(parents=True, exist_ok=True)
                draw_overlay(image, data).save(args.out)
            else:
                from .runner import FrozenSAM2
                runner = FrozenSAM2(args.config, args.checkpoint, args.device, args.precision)
                args.out.mkdir(parents=True, exist_ok=True)
                failed = False
                for index, image_path in enumerate(args.image):
                    # 序号避免批量输入中同名图片互相覆盖。
                    stem = f"{index:04d}_{image_path.stem}"
                    image = read_rgb(image_path)
                    mask, metadata = runner.segment(image, data, args.min_score)
                    metadata.update({"image": str(image_path.resolve()), "target": data["target"],
                                     "prompts": str(args.prompts.resolve())})
                    metadata.update(save_segmentation_artifacts(args.out, stem, image, data, mask))
                    if mask is None:
                        failed = True
                    save_json(args.out / f"{stem}.json", metadata)
                    print(f"{image_path.name}: {metadata['status']}")
                    if "mask_path" in metadata:
                        print(f"  mask: {metadata['mask_path']}")
                return 2 if failed else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
