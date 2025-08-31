#!/usr/bin/env python3
"""
Simple CLI runner for the Anomaly Detection project.
"""

import argparse
import sys
from pathlib import Path

def create_mvtec_structure():
    """Create MVTec dataset folder structure (empty - requires manual download)."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create MVTec folder structure only (no content)
    mvtec_dir = data_dir / "mvtec"
    mvtec_dir.mkdir(exist_ok=True)
    
    # Create class directories
    for class_name in ["carpet", "grid"]:
        class_dir = mvtec_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create train/test directories
        (class_dir / "train").mkdir(exist_ok=True)
        (class_dir / "test").mkdir(exist_ok=True)
        (class_dir / "ground_truth").mkdir(exist_ok=True)
    
    print("✅ Created MVTec folder structure")
    print("📁 Note: Dataset must be downloaded manually from https://www.mvtec.com/company/research/datasets/mvtec-ad/")
    return True

def setup_project():
    """Setup the project structure."""
    print("🔧 Setting up project structure...")
    
    # Create necessary directories
    dirs = ["results", "logs", "checkpoints"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✅ Created {dir_name}/")
    
    # Create MVTec folder structure
    create_mvtec_structure()
    
    print("✅ Project setup complete!")
    return True

def show_help():
    """Show help information."""
    help_text = """
🚀 Anomaly Detection Super-Resolution Project

Available commands:
  setup     - Initialize project structure and sample data
  test      - Run basic tests to verify setup
  help      - Show this help message

Examples:
  python run.py setup
  python run.py test
  python run.py help

For training (when implemented):
  python src/main.py --dataset mvtec --class-name carpet --model drn-l
"""
    print(help_text)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Super-Resolution CLI",
        add_help=False
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "test", "help"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_project()
    elif args.command == "test":
        print("🧪 Running tests...")
        try:
            import test_basic
            test_basic.main()
        except ImportError:
            print("❌ Test script not found. Run 'python run.py setup' first.")
    elif args.command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {args.command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_help()
    else:
        main()
