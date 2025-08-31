#!/usr/bin/env python3
"""
Prepare MVTec AD dataset for anomaly detection training and evaluation.

This script:
1. Resizes all images to 128x128 (manageable for local training)
2. Creates LR/HR pairs for super-resolution training
3. Organizes test data into good/bad structure for evaluation
4. Works with the current folder structure without modifying originals
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

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

def process_training_data(source_dir, target_dir, scale_factor=4):
    """Process training data: resize to 128x128 and create LR versions."""
    print(f"📁 Processing training data: {source_dir.name}")
    
    # Create target directories
    hr_dir = target_dir / "good" / "HR"
    lr_dir = target_dir / "good" / f"LR_{scale_factor}"
    
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    image_files = list(source_dir.glob("*.png"))
    print(f"  Found {len(image_files)} training images")
    
    for img_file in tqdm(image_files, desc="Processing training images"):
        # Resize to 128x128
        hr_128 = resize_image(img_file, target_size=(128, 128))
        
        # Create LR version (32x32 for 4x scale)
        lr_32 = create_lr_image(hr_128, scale_factor=scale_factor)
        
        # Save both versions
        hr_path = hr_dir / img_file.name
        lr_path = lr_dir / img_file.name
        
        save_image_pair(hr_128, lr_32, hr_path, lr_path)
    
    print(f"  ✅ Created {len(image_files)} HR/LR pairs")

def process_test_data(source_dir, target_dir, scale_factor=4):
    """Process test data: organize into good/bad structure."""
    print(f"📁 Processing test data: {source_dir.name}")
    
    # Create target directories
    good_hr_dir = target_dir / "good" / "HR"
    good_lr_dir = target_dir / "good" / f"LR_{scale_factor}"
    bad_hr_dir = target_dir / "bad" / "HR"
    bad_lr_dir = target_dir / "bad" / f"LR_{scale_factor}"
    
    for dir_path in [good_hr_dir, good_lr_dir, bad_hr_dir, bad_lr_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process good images
    good_source = source_dir / "good"
    if good_source.exists():
        good_images = list(good_source.glob("*.png"))
        print(f"  Processing good: {len(good_images)} images")
        
        for img_file in tqdm(good_images, desc="  good", leave=False):
            hr_128 = resize_image(img_file, target_size=(128, 128))
            lr_32 = create_lr_image(hr_128, scale_factor=scale_factor)
            
            hr_path = good_hr_dir / img_file.name
            lr_path = good_lr_dir / img_file.name
            save_image_pair(hr_128, lr_32, hr_path, lr_path)
    
    # Process all anomaly images (combine into bad)
    anomaly_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name != "good"]
    total_bad = 0
    
    for anomaly_dir in anomaly_dirs:
        anomaly_images = list(anomaly_dir.glob("*.png"))
        print(f"  Processing {anomaly_dir.name}: {len(anomaly_images)} images")
        
        for img_file in tqdm(anomaly_images, desc=f"    {anomaly_dir.name}", leave=False):
            hr_128 = resize_image(img_file, target_size=(128, 128))
            lr_32 = create_lr_image(hr_128, scale_factor=scale_factor)
            
            # Create unique filename to avoid conflicts
            new_name = f"{anomaly_dir.name}_{img_file.name}"
            hr_path = bad_hr_dir / new_name
            lr_path = bad_lr_dir / new_name
            
            save_image_pair(hr_128, lr_32, hr_path, lr_path)
            total_bad += 1
    
    # Count final results
    good_count = len(list(good_hr_dir.glob("*.png")))
    bad_count = len(list(bad_hr_dir.glob("*.png")))
    
    print(f"  ✅ Good test images: {good_count}")
    print(f"  ✅ Bad test images: {bad_count}")

def prepare_mvtec_dataset(source_base="data/mvtec", target_base="data/mvtec_128", scale_factor=4):
    """Prepare the complete MVTec dataset for 128x128 training."""
    print("🚀 Preparing MVTec AD dataset for 128x128 training")
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
        test_target = target_base / class_name / "test"
        
        # Process training data
        if train_source.exists():
            process_training_data(train_source, train_target, scale_factor)
        else:
            print(f"  ❌ Training data not found: {train_source}")
        
        # Process test data
        if test_source.exists():
            process_test_data(test_source, test_target, scale_factor)
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
        train_lr = base_path / class_name / "train" / "good" / "LR_4"
        
        if train_hr.exists():
            train_count = len(list(train_hr.glob("*.png")))
            print(f"    ✅ train/good/HR: {train_count} images")
        else:
            print(f"    ❌ train/good/HR: missing")
            
        if train_lr.exists():
            lr_count = len(list(train_lr.glob("*.png")))
            print(f"    ✅ train/good/LR_4: {lr_count} images")
        else:
            print(f"    ❌ train/good/LR_4: missing")
        
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
    print("🎯 MVTec AD Dataset Preparation for 128x128 Training (v2)")
    print("=" * 60)
    
    # Check if source data exists
    source_base = Path("data/mvtec")
    if not source_base.exists():
        print("❌ Source data not found. Please ensure MVTec dataset is in data/mvtec/")
        return
    
    # Prepare dataset
    prepare_mvtec_dataset(
        source_base="data/mvtec",
        target_base="data/mvtec_128",
        scale_factor=4
    )
    
    # Verify structure
    verify_dataset_structure("data/mvtec_128")
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📝 Next steps:")
    print(f"   1. Update main.py to use data/mvtec_128/")
    print(f"   2. Test the training pipeline")
    print(f"   3. Verify data loading works correctly")

if __name__ == "__main__":
    main()
