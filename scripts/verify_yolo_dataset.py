#!/usr/bin/env python3
"""
Verify YOLO dataset integrity.
"""

import os
import sys
import argparse

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sam3_annotation_tool.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO dataset integrity")
    parser.add_argument("dataset_path", nargs='?', help="Path to dataset root")
    
    args = parser.parse_args()
    
    # Default path if not provided
    if not args.dataset_path:
        args.dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", 
            "GOB_SAM3_dataset"
        )
    
    if not os.path.exists(args.dataset_path):
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        sys.exit(1)

    manager = DatasetManager(args.dataset_path)
    
    try:
        success = manager.verify_dataset()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
