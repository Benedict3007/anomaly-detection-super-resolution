#!/usr/bin/env python3
"""
Basic test script to verify the project setup works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test basic imports."""
    try:
        import torch
        print(f"✅ PyTorch imported successfully: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy imported successfully: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib imported successfully: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_paths():
    """Test that basic project structure exists."""
    project_root = Path(__file__).parent
    
    required_dirs = ["src", "data", "results"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
            return False
    
    return True

def test_main_script():
    """Test that main.py can be imported."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from main import parse_args, setup_data_paths
        print("✅ Main script imports successfully")
        return True
    except Exception as e:
        print(f"❌ Main script import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running basic project tests...\n")
    
    tests = [
        ("Basic imports", test_imports),
        ("Project structure", test_paths),
        ("Main script", test_main_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed\n")
        else:
            print(f"❌ {test_name} failed\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic setup is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
