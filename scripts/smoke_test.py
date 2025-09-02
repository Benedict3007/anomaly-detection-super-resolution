#!/usr/bin/env python3
"""
Smoke test for fresh clones:
1) Prepare MVTec 'grid' at 128x128 HR with scale 4 (and LR_2 for pipeline compatibility)
2) Train DRCT for 5 epochs on the prepared data
3) Evaluate anomaly detection AUC on the test split using the trained model

Usage:
  python scripts/smoke_test.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_source_dataset() -> bool:
    mvtec_root = PROJECT_ROOT / "data" / "mvtec"
    grid_train = mvtec_root / "grid" / "train" / "good"
    grid_test = mvtec_root / "grid" / "test"
    if not mvtec_root.exists():
        print("ERROR: data/mvtec/ not found. Please download MVTec AD dataset as described in README.md.")
        return False
    if not grid_train.exists() or not grid_test.exists():
        print("ERROR: MVTec grid class not found at expected paths under data/mvtec/grid/.")
        return False
    return True


def prepare_grid_128x_scale4() -> Path:
    """Prepare only 'grid' class at 128 HR and scale 4 using existing helpers."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.prepare_mvtec_data import (
            process_training_data,
            process_test_data,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import dataset preparation helpers: {e}")

    source_base = PROJECT_ROOT / "data" / "mvtec"
    target_base = PROJECT_ROOT / "data" / "mvtec_128"
    class_name = "grid"

    # Clean only the target class directory to keep this quick
    target_class_dir = target_base / class_name
    if target_class_dir.exists():
        shutil.rmtree(target_class_dir)

    # Compute paths
    train_source = source_base / class_name / "train" / "good"
    test_source = source_base / class_name / "test"
    train_target = target_base / class_name / "train"
    val_target = target_base / class_name / "val"
    test_target = target_base / class_name / "test"

    # Prepare
    print("Preparing MVTec grid at 128x128 with scale=4 ...")
    process_training_data(
        train_source,
        train_target,
        val_target,
        scale_factors=(4,),
        target_hr=(128, 128),
        val_ratio=0.1,
        seed=42,
    )
    process_test_data(
        test_source,
        test_target,
        scale_factors=(4,),
        target_hr=(128, 128),
    )

    print("Dataset preparation complete for grid.")
    return target_base


def run_training(data_root: Path, epochs: int = 5) -> Optional[Path]:
    """Run DRCT training and return the created run directory if successful."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--model-type", "drct",
        "--dataset", "mvtec",
        "--classe", "grid",
        "--resolution", "128",
        "--scale", "4",
        "--epochs", str(epochs),
        "--data-root", str(data_root),
        "--save-dir", str(PROJECT_ROOT / "workspace" / "experiment"),
        "--workers", "0",
    ]

    print("Starting training (DRCT, 5 epochs)...")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"ERROR: Training failed with exit code {rc}")
        return None

    # Find the latest DRCT run for grid at 128 and X4
    runs_root = PROJECT_ROOT / "workspace" / "experiment" / "drct"
    candidates = []
    if runs_root.exists():
        for p in runs_root.iterdir():
            if p.is_dir() and p.name.startswith("mvtec_grid_128_X4"):
                candidates.append((p.stat().st_mtime, p))
    if not candidates:
        print("ERROR: Could not locate the training run directory.")
        return None
    latest = sorted(candidates, key=lambda x: x[0])[-1][1]
    print(f"Training complete. Run directory: {latest}")
    return latest


def run_evaluation(run_dir: Path) -> int:
    """Evaluate the trained model on test set and print AUCs.
    Evaluation infers dataset/class/resolution/scale from run-dir/config.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "evaluate.py"),
        "--run-dir", str(run_dir),
        "--save-images",
    ]
    print("Evaluating on test set (computing AUCs)...")
    return subprocess.call(cmd)


def main() -> int:
    print("=== Anomaly Detection Super-Resolution: Smoke Test ===")
    if not check_source_dataset():
        return 1

    data_root = prepare_grid_128x_scale4()
    run_dir = run_training(data_root, epochs=5)
    if run_dir is None:
        return 1

    rc = run_evaluation(run_dir)
    if rc != 0:
        print(f"ERROR: Evaluation failed with exit code {rc}")
        return rc

    print("Smoke test finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


