import argparse
import json
import time
import sys
from pathlib import Path

import yaml

from pipeline.preprocessor import preprocess
from pipeline.bw_sketch import generate_bw_sketch
from pipeline.flat_color import generate_flat_color
from pipeline.vectorizer import vectorize
from pipeline.quality import compute_metrics, generate_comparison


def load_config(preset: str = "default") -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    presets = cfg.get("presets", {})
    return presets.get(preset, presets.get("default", {}))


def detect_preset(filename: str) -> str:
    name = filename.lower()
    if "sneaker" in name or "shoe" in name or "footwear" in name:
        return "footwear"
    if "gem" in name or "stone" in name:
        return "jewelry_gem"
    return "jewelry_metal"


def process_single(image_path: Path, output_dir: Path, config: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    data = preprocess(str(image_path), config)
    bw_raster = generate_bw_sketch(data, config)
    color_raster = generate_flat_color(data, config)
    svg_content, stats = vectorize(bw_raster, color_raster, data, config)
    elapsed = time.time() - start

    import cv2
    cv2.imwrite(str(output_dir / "bw_sketch.png"), bw_raster)
    cv2.imwrite(str(output_dir / "flat_color.png"), color_raster)

    with open(output_dir / "result.svg", "w", encoding="utf-8") as f:
        f.write(svg_content)

    metrics = compute_metrics(data, bw_raster, color_raster, svg_content, stats, elapsed)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    comparison = generate_comparison(data["bgr"], bw_raster, color_raster, data["alpha"])
    cv2.imwrite(str(output_dir / "comparison.png"), comparison)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Flat Sketch Normalization & Vectorization Pipeline"
    )
    parser.add_argument("input", help="Input PNG file or directory of PNGs")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--preset", default=None, help="Config preset (jewelry_metal, jewelry_gem, footwear, default)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_base = Path(args.output_dir)

    is_single = input_path.is_file()
    if is_single:
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(input_path.glob("*.png"))
        if not images:
            print(f"No PNG files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)

    all_metrics = {}
    for img_path in images:
        preset = args.preset or detect_preset(img_path.stem)
        config = load_config(preset)
        out_dir = output_base if is_single else output_base / img_path.stem
        print(f"Processing {img_path.name} (preset: {preset})...")

        try:
            metrics = process_single(img_path, out_dir, config)
            all_metrics[img_path.name] = metrics
            print(f"  IoU: {metrics['silhouette_iou']:.4f} | "
                  f"F-score: {metrics['boundary_f_score']:.4f} | "
                  f"Fills: {metrics['fill_count']} | "
                  f"Anchors: {metrics['total_anchors']} | "
                  f"Time: {metrics['processing_time_s']:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics[img_path.name] = {"error": str(e)}

    if len(images) > 1:
        with open(output_base / "aggregate_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nProcessed {len(images)} images. Results in {output_base}/")


if __name__ == "__main__":
    main()
