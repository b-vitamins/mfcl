import argparse
import os
from pathlib import Path
import random


def _scan_split(root: Path, split: str, exts):
    split_dir = root / split
    if not split_dir.exists():
        return []
    out = []
    for dp, _, files in os.walk(split_dir):
        for f in files:
            p = Path(dp) / f
            if p.suffix.lower() in exts:
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Dataset root containing 'train' and/or 'val' folders.",
    )
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--relative", action="store_true", help="Write paths relative to root."
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle output deterministically with --seed.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", nargs="+", default=[".jpeg", ".jpg", ".png"])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split != "both" else ["train", "val"]
    for split in splits:
        files = _scan_split(root, split, set(e.lower() for e in args.exts))
        files = sorted(files)
        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(files)
        if args.relative:
            lines = [str(p.relative_to(root)) for p in files]
        else:
            lines = [str(p) for p in files]
        outp = outdir / f"imagenet_{split}.txt"
        with open(outp, "w") as f:
            f.write("\n".join(lines))
        print(f"[{split}] count={len(files)} -> {outp}")
        if len(lines) > 0:
            print(" first3:", lines[:3])
            print(" last3:", lines[-3:])


if __name__ == "__main__":
    main()
