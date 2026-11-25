"""Batch preprocessing script to process Candling dataset folders.

Author: Anas (Image Processing Engineer - Track 1)
Usage:
    python src/data/batch_preprocess.py --input_dir "Egg Fertility" --output_dir "processed_dataset" --img_size 640
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2

from preprocessor import CandlingPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Candling Preprocessing Batch Script (Anas)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Egg Fertility",
        help="Path to raw candling dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_preprocessed",
        help="Path to save preprocessed images"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Target square image resolution (e.g. 640 or 256)"
    )
    parser.add_argument(
        "--no_green",
        action="store_true",
        help="Disable G-channel extraction (use standard grayscale instead)"
    )
    parser.add_argument(
        "--no_clahe",
        action="store_true",
        help="Disable CLAHE illumination normalization"
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="Output 1-channel images instead of duplicating to 3 channels"
    )
    return parser.parse_args()


def process_dataset(
    input_dir: str,
    output_dir: str,
    img_size: int = 640,
    extract_green: bool = True,
    apply_clahe: bool = True,
    return_3channel: bool = True,
):
    preprocessor = CandlingPreprocessor(
        target_size=(img_size, img_size),
        extract_green=extract_green,
        apply_clahe=apply_clahe,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [
        f for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    print(f"Found {len(image_files)} images in '{input_dir}'")
    print(f"Target size: {img_size}x{img_size} | Extract G-channel: {extract_green} | CLAHE: {apply_clahe}")
    print(f"Writing outputs to: '{output_dir}'")

    for file in tqdm(image_files, desc="Preprocessing Images"):
        relative = file.relative_to(input_path)
        dest_file = output_path / relative
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(file))
        if img is None:
            print(f"Warning: Could not read {file}, skipping.")
            continue

        processed = preprocessor.process(img, return_3channel=return_3channel)
        cv2.imwrite(str(dest_file), processed)

    print(f"Batch preprocessing completed successfully! Total processed: {len(image_files)}")


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        extract_green=not args.no_green,
        apply_clahe=not args.no_clahe,
        return_3channel=not args.single_channel,
    )
