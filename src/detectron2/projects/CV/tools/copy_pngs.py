#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy all .png files from one folder to another.

Default (matches this repo layout):
  src: detectron2/plug_train_linux_2
  dst: detectron2/plug_train2

Examples:
  python detectron2/tools/copy_pngs.py
  python detectron2/tools/copy_pngs.py --overwrite
  python detectron2/tools/copy_pngs.py --dry-run
  python detectron2/tools/copy_pngs.py --src /abs/src --dst /abs/dst
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]  # .../detectron2
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(repo_root / "plug_train_linux_2"), help="source directory")
    p.add_argument("--dst", default=str(repo_root / "plug_train2"), help="destination directory")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="print what would be copied without copying")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"src directory not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print(f"[copy_pngs] No .png files found in: {src}")
        return

    copied = 0
    skipped = 0
    for p in pngs:
        out = dst / p.name
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"[dry-run] {p} -> {out}")
        else:
            shutil.copy2(p, out)
        copied += 1

    print(
        f"[copy_pngs] src={src} dst={dst} total_png={len(pngs)} "
        f"copied={copied} skipped={skipped} overwrite={bool(args.overwrite)} dry_run={bool(args.dry_run)}"
    )


if __name__ == "__main__":
    # Avoid issues with some environments that set a restrictive umask
    os.umask(0o022)
    main()


