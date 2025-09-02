#!/usr/bin/env python3
"""
Prepare MVTec AD dataset for anomaly detection training and evaluation.

This script:
1. Resizes all images to a chosen HR size (256/128/64/32)
2. Creates LR images for chosen downscales (4 and/or 8)
   - Progressive LR mode (for DRN-L): also creates LR_2 and intermediate levels
     e.g., for scale 4 → LR_2 and LR_4; for scale 8 → LR_2, LR_4, LR_8
3. Organizes test data into good/bad structure for evaluation
4. Works with the current folder structure without modifying originals
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]
import argparse

def resize_image(image_path, target_size=(128, 128), resample=Image.LANCZOS):
    """Resize image to target size."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img.resize(target_size, resample)

def create_lr_image(hr_image, scale_factor=4, resample=Image.LANCZOS):
    """Create low-resolution image from high-resolution image."""
    lr_size = (hr_image.size[0] // scale_factor, hr_image.size[1] // scale_factor)
    return hr_image.resize(lr_size, resample)

def save_image_pair(hr_image, lr_image, hr_path, lr_path):
    """Save HR and LR image pair."""
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    
    hr_image.save(hr_path)
    lr_image.save(lr_path)

def process_training_data(source_dir, train_target_dir, val_target_dir, scale_factors=(4,), target_hr=(128, 128), val_ratio=0.1, seed=42):
    """Process training data: create train/val splits, resize to target_hr, and create LR versions."""
    print(f"📁 Processing training data: {source_dir.name}")

    # Prepare directory maps for train and val
    def make_dirs(base_dir):
        dirs = {
            'hr': base_dir / "good" / "HR",
            'lr': {}
        }
        dirs['hr'].mkdir(parents=True, exist_ok=True)
        for s in scale_factors:
            p = base_dir / "good" / f"LR_{s}"
            p.mkdir(parents=True, exist_ok=True)
            dirs['lr'][s] = p
        return dirs

    train_dirs = make_dirs(train_target_dir)
    val_dirs = make_dirs(val_target_dir)

    # Get and split image files
    image_files = list(source_dir.glob("*.png"))
    print(f"  Found {len(image_files)} training images")
    if len(image_files) == 0:
        print("  ⚠️  No training images found. Skipping train/val split.")
        return

    rng = np.random.RandomState(seed)
    rng.shuffle(image_files)
    val_size = int(len(image_files) * float(val_ratio))
    val_size = max(1, val_size) if len(image_files) > 1 and val_ratio > 0 else 0
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]

    def save_split(files, dirs, desc):
        for img_file in tqdm(files, desc=desc):
            hr_img = resize_image(img_file, target_size=target_hr)
            hr_path = dirs['hr'] / img_file.name
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            hr_img.save(hr_path)

            for s in scale_factors:
                lr_img = create_lr_image(hr_img, scale_factor=s)
                lr_path = dirs['lr'][s] / img_file.name
                lr_img.save(lr_path)

    save_split(train_files, train_dirs, "Processing train images")
    if val_size > 0:
        save_split(val_files, val_dirs, "Processing val images")

    print(f"  ✅ Created {len(train_files)} train pairs and {len(val_files)} val pairs")

def process_test_data(source_dir, target_dir, scale_factors=(4,), target_hr=(128, 128)):
    """Process test data: organize into good/bad structure for all requested scales at target_hr size."""
    print(f"📁 Processing test data: {source_dir.name}")
    
    # Create target directories
    good_hr_dir = target_dir / "good" / "HR"
    bad_hr_dir = target_dir / "bad" / "HR"
    
    dirs_to_make = [good_hr_dir, bad_hr_dir]
    good_lr_dirs = {}
    bad_lr_dirs = {}
    for s in scale_factors:
        good_lr_dirs[s] = target_dir / "good" / f"LR_{s}"
        bad_lr_dirs[s] = target_dir / "bad" / f"LR_{s}"
        dirs_to_make.extend([good_lr_dirs[s], bad_lr_dirs[s]])

    for dir_path in dirs_to_make:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process good images
    good_source = source_dir / "good"
    if good_source.exists():
        good_images = list(good_source.glob("*.png"))
        print(f"  Processing good: {len(good_images)} images")
        
        for img_file in tqdm(good_images, desc="  good", leave=False):
            hr_img = resize_image(img_file, target_size=target_hr)
            hr_path = good_hr_dir / img_file.name
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            hr_img.save(hr_path)

            for s in scale_factors:
                lr_img = create_lr_image(hr_img, scale_factor=s)
                lr_path = good_lr_dirs[s] / img_file.name
                lr_img.save(lr_path)
    
    # Process all anomaly images (combine into bad)
    anomaly_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name != "good"]
    total_bad = 0
    
    for anomaly_dir in anomaly_dirs:
        anomaly_images = list(anomaly_dir.glob("*.png"))
        print(f"  Processing {anomaly_dir.name}: {len(anomaly_images)} images")
        
        for img_file in tqdm(anomaly_images, desc=f"    {anomaly_dir.name}", leave=False):
            hr_img = resize_image(img_file, target_size=target_hr)
            
            # Create unique filename to avoid conflicts
            new_name = f"{anomaly_dir.name}_{img_file.name}"
            hr_path = bad_hr_dir / new_name
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            hr_img.save(hr_path)

            for s in scale_factors:
                lr_img = create_lr_image(hr_img, scale_factor=s)
                lr_path = bad_lr_dirs[s] / new_name
                lr_img.save(lr_path)
            total_bad += 1
    
    # Count final results
    good_count = len(list(good_hr_dir.glob("*.png")))
    bad_count = len(list(bad_hr_dir.glob("*.png")))
    
    print(f"  ✅ Good test images: {good_count}")
    print(f"  ✅ Bad test images: {bad_count}")

def prepare_mvtec_dataset(source_base="data/mvtec", target_base="data/mvtec_128", scale_factors=(4,), target_hr=(128, 128), val_ratio=0.1, seed=42):
    """Prepare the complete MVTec dataset for chosen HR size with selected LR scales and a train/val split."""
    print(f"🚀 Preparing MVTec AD dataset for {target_hr[0]}x{target_hr[1]} training")
    print("=" * 60)
    
    source_base = Path(source_base)
    target_base = Path(target_base)
    
    # Remove existing target directory
    if target_base.exists():
        shutil.rmtree(target_base)
        print("🧹 Cleaned existing target directory")
    
    # Process each class
    classes = ["carpet", "grid"]
    
    for class_name in classes:
        print(f"\n🔧 Processing class: {class_name}")
        
        # Source paths
        train_source = source_base / class_name / "train" / "good"
        test_source = source_base / class_name / "test"
        
        # Target paths
        train_target = target_base / class_name / "train"
        val_target = target_base / class_name / "val"
        test_target = target_base / class_name / "test"
        
        # Process training data
        if train_source.exists():
            process_training_data(train_source, train_target, val_target, scale_factors, target_hr=target_hr, val_ratio=val_ratio, seed=seed)
        else:
            print(f"  ❌ Training data not found: {train_source}")
        
        # Process test data
        if test_source.exists():
            process_test_data(test_source, test_target, scale_factors, target_hr=target_hr)
        else:
            print(f"  ❌ Test data not found: {test_source}")
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Output directory: {target_base}")
    print(f"🔍 Check the structure with: find {target_base} -type d | sort")

def verify_dataset_structure(base_dir):
    """Verify the prepared dataset structure."""
    print(f"\n🔍 Verifying dataset structure: {base_dir}")
    
    base_path = Path(base_dir)
    classes = ["carpet", "grid"]
    
    for class_name in classes:
        print(f"\n  📁 {class_name}/")
        
        # Check training data
        train_hr = base_path / class_name / "train" / "good" / "HR"
        if train_hr.exists():
            train_count = len(list(train_hr.glob("*.png")))
            print(f"    ✅ train/good/HR: {train_count} images")
        else:
            print(f"    ❌ train/good/HR: missing")
        # List LR_* present and counts
        lr_dirs = sorted((base_path / class_name / "train" / "good").glob("LR_*"))
        for lr_dir in lr_dirs:
            lr_count = len(list(lr_dir.glob("*.png")))
            print(f"    ✅ train/good/{lr_dir.name}: {lr_count} images")
        
        # Check validation data
        val_hr = base_path / class_name / "val" / "good" / "HR"
        if val_hr.exists():
            val_count = len(list(val_hr.glob("*.png")))
            print(f"    ✅ val/good/HR: {val_count} images")
        else:
            print(f"    ❌ val/good/HR: missing")
        lr_val_dirs = sorted((base_path / class_name / "val" / "good").glob("LR_*"))
        for lr_dir in lr_val_dirs:
            lr_count = len(list(lr_dir.glob("*.png")))
            print(f"    ✅ val/good/{lr_dir.name}: {lr_count} images")

        # Check test data
        test_good_hr = base_path / class_name / "test" / "good" / "HR"
        test_bad_hr = base_path / class_name / "test" / "bad" / "HR"
        
        if test_good_hr.exists():
            good_count = len(list(test_good_hr.glob("*.png")))
            print(f"    ✅ test/good/HR: {good_count} images")
        else:
            print(f"    ❌ test/good/HR: missing")
            
        if test_bad_hr.exists():
            bad_count = len(list(test_bad_hr.glob("*.png")))
            print(f"    ✅ test/bad/HR: {bad_count} images")
        else:
            print(f"    ❌ test/bad/HR: missing")
    
    print("✅ Dataset verification complete!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MVTec AD dataset preparation")
    parser.add_argument("--hr-size", type=int, default=128, choices=[256, 128, 64, 32])
    parser.add_argument("--scales", type=str, default="4", help="Comma-separated downscale factors: 4,8")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"🎯 MVTec AD Dataset Preparation ({args.hr_size}x{args.hr_size})")
    print("=" * 60)

    # Check if source data exists
    source_base = Path("data/mvtec")
    if not source_base.exists():
        print("❌ Source data not found. Please ensure MVTec dataset is in data/mvtec/")
        return

    # Compute scale factors
    try:
        user_scales = sorted({int(s.strip()) for s in args.scales.split(',') if s.strip()})
    except ValueError:
        print("❌ Invalid --scales. Use comma-separated integers from {4,8}")
        return
    for s in user_scales:
        if s not in (4, 8):
            print("❌ Only scales 4 and/or 8 are supported")
            return
    scales = set(user_scales)
    # Progressive LR is permanent: always include LR_2; if 8 requested, ensure LR_4 too
    scales.add(2)
    if 8 in user_scales:
        scales.add(4)
    scale_factors = tuple(sorted(scales))

    target_hr = (args.hr_size, args.hr_size)
    target_base = Path(f"data/mvtec_{args.hr_size}")

    # Prepare dataset
    prepare_mvtec_dataset(
        source_base="data/mvtec",
        target_base=str(target_base),
        scale_factors=scale_factors,
        target_hr=target_hr,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Verify structure
    verify_dataset_structure(str(target_base))
    
    print(f"\n🎉 Dataset preparation complete!")

if __name__ == "__main__":
    main()
