import argparse
from pathlib import Path
from manhwa_cropper.pipeline import crop_page_to_scenes

def iter_images(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in exts:
            yield p

def main():
    ap = argparse.ArgumentParser("manhwa-cropper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    crop = sub.add_parser("crop", help="Crop manhwa/webtoon pages into scene images.")
    crop.add_argument("--input", type=str, default=None, help="Directory of pages.")
    crop.add_argument("--image", type=str, default=None, help="Single page image path.")
    crop.add_argument("--output", type=str, required=True, help="Output directory.")
    crop.add_argument("--min-scene-height", type=int, default=220, help="Minimum scene height in pixels.")
    crop.add_argument("--min-gutter-height", type=int, default=18, help="Minimum gutter band height in pixels.")
    crop.add_argument("--max-scenes", type=int, default=120, help="Safety cap per page.")
    crop.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size.")
    crop.add_argument("--conf", type=float, default=0.25, help="YOLO confidence.")
    crop.add_argument("--iou", type=float, default=0.5, help="YOLO IoU for NMS.")
    crop.add_argument("--trim", action="store_true", help="Enable smart trim (recommended).")
    crop.add_argument("--keep-json", action="store_true", help="Write JSON metadata.")
    crop.add_argument("--device", type=str, default="cpu", help="cpu or 0/1 for GPU device id.")

    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.image:
        pages = [Path(args.image)]
    elif args.input:
        pages = list(iter_images(Path(args.input)))
    else:
        raise SystemExit("Provide --image or --input")
    print(f"Found {len(pages)} image(s) to process.")

    for p in pages:
        crop_page_to_scenes(
            image_path=p,
            out_dir=outdir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            min_scene_h=args.min_scene_height,
            min_gutter_h=args.min_gutter_height,
            max_scenes=args.max_scenes,
            enable_trim=args.trim,
            write_json=args.keep_json,
        )

if __name__ == "__main__":
    main()
