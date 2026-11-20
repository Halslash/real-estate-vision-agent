"""Reproducible qualitative demonstration on real property images.

Scores real images with a saved checkpoint and writes an HTML page next to a CSV:
each entry shows the ORIGINAL image, the predicted class, the full low/average/
high probability vector, the normalized condition_score and an explicitly
labelled confidence value (max predicted class probability, NOT calibrated).

This is a qualitative demo, not held-out evidence: images are scored as supplied.
Use test-split images if the demo is meant to illustrate generalization.

Example (Kaggle):
    !python src/demonstrate.py \
        --checkpoint /kaggle/working/output/improved_mobilenet/models/best_model.pth \
        --images_dir /kaggle/input/.../socal_pics --num_examples 6 \
        --output_dir /kaggle/working/output/demonstration
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CSV_COLUMNS = (
    "image_id",
    "image",
    "predicted_class",
    "p_low",
    "p_average",
    "p_high",
    "condition_score",
    "confidence",
    "model",
    "score_type",
)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Self-contained checkpoint produced by src/train.py.")
    parser.add_argument("--image_paths", nargs="*", default=[],
                        help="Explicit real image paths (takes priority over --images_dir).")
    parser.add_argument("--images_dir", default=None,
                        help="Directory of real dataset images to sample when no paths are given.")
    parser.add_argument("--num_examples", type=int, default=6,
                        help="How many images to sample from --images_dir (sorted order).")
    parser.add_argument("--output_dir", default="/kaggle/working/output/demonstration")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def collect_images(args: argparse.Namespace) -> List[str]:
    """Resolve real image paths without decoding pixels or inventing examples."""
    paths = [os.fspath(path) for path in args.image_paths]
    if not paths:
        if not args.images_dir:
            raise ValueError("Provide --image_paths or --images_dir with real images.")
        if args.num_examples < 1:
            raise ValueError("--num_examples must be at least 1.")
        entries = sorted(
            os.path.join(args.images_dir, name)
            for name in os.listdir(args.images_dir)
            if name.lower().endswith(_IMAGE_EXTS)
        )
        paths = entries[:args.num_examples]
    if not paths:
        raise ValueError("No real images were found; refusing to create an empty demo.")
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return paths


def build_rows(paths: List[str], results: List[Dict[str, Any]], image_dir: str) -> List[Dict[str, Any]]:
    """Join each original image with its scoring result; no fabricated fields."""
    rows: List[Dict[str, Any]] = []
    for index, (path, result) in enumerate(zip(paths, results)):
        filename = f"{index:03d}_{os.path.basename(path)}"
        shutil.copy2(path, os.path.join(image_dir, filename))
        probabilities = result["probabilities"]
        rows.append({
            "image_id": os.path.splitext(os.path.basename(path))[0],
            "image": os.path.join("images", filename),
            "predicted_class": result["label"],
            "p_low": float(probabilities["low"]),
            "p_average": float(probabilities["average"]),
            "p_high": float(probabilities["high"]),
            "condition_score": float(result["condition_score"]),
            "confidence": float(result["confidence"]),
            "model": result["model"],
            "score_type": result["score_type"],
        })
    return rows


def build_html(rows: List[Dict[str, Any]], checkpoint: str) -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Member A visual proxy demonstration</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;"
        "color:#222}img{max-width:420px;max-height:320px;display:block}table{border-collapse:"
        "collapse;margin:.6em 0}td,th{border:1px solid #ccc;padding:.35em .7em;text-align:left}"
        "h2{margin-top:2em}.warn{background:#fff4cc;padding:.8em;border-left:4px solid #d99}</style>",
        "</head><body>",
        "<h1>Member A — visual proxy demonstration (real images)</h1>",
        "<p class='warn'><strong>Interpretation warning:</strong> this is a market-derived "
        "visual proxy, not verified physical condition, not an aesthetic grade, not a price. "
        "This page is a qualitative demo, not held-out evidence.</p>",
        f"<p>Checkpoint: <code>{html.escape(checkpoint)}</code></p>",
    ]
    for row in rows:
        parts.extend([
            "<hr>",
            f"<h2>{html.escape(str(row['image_id']))}</h2>",
            f"<img src='{html.escape(row['image'])}' alt='Original {html.escape(str(row['image_id']))}'>",
            "<table>",
            f"<tr><th>Predicted class (argmax)</th><td>{html.escape(str(row['predicted_class']))}</td></tr>",
            f"<tr><td>P(low)</td><td>{row['p_low']:.4f}</td></tr>",
            f"<tr><td>P(average)</td><td>{row['p_average']:.4f}</td></tr>",
            f"<tr><td>P(high)</td><td>{row['p_high']:.4f}</td></tr>",
            f"<tr><th>Normalized condition_score (0.5·P(average) + P(high))</th>"
            f"<td>{row['condition_score']:.4f}</td></tr>",
            f"<tr><th>Confidence = max predicted class probability (NOT calibrated)</th>"
            f"<td>{row['confidence']:.4f}</td></tr>",
            f"<tr><td>Model</td><td>{html.escape(str(row['model']))}</td></tr>",
            f"<tr><td>Score type</td><td>{html.escape(str(row['score_type']))}</td></tr>",
            "</table>",
        ])
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # Lazy import: --help and static checks work without torch installed.
    from src.vision_agent import VisionAgent  # noqa: E402

    paths = collect_images(args)
    print(f"Scoring {len(paths)} real images with {args.checkpoint}")
    agent = VisionAgent(checkpoint_path=args.checkpoint, device=args.device)
    results = agent.score_batch(paths, batch_size=args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    rows = build_rows(paths, results, image_dir)

    csv_path = os.path.join(args.output_dir, "demonstration.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    html_path = os.path.join(args.output_dir, "demonstration.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(build_html(rows, args.checkpoint))
    print(f"Wrote {len(rows)} demonstrations -> {csv_path}, {html_path}")


if __name__ == "__main__":
    main()
