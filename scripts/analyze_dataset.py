#!/usr/bin/env python3
"""
Analyze YOLO dataset polygons to determine optimal cleanup parameters.
"""

import os
import sys
import argparse
import json

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sam3_annotation_tool.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset polygons")
    parser.add_argument("dataset_path", nargs='?', help="Path to dataset root")
    parser.add_argument("--output-report", help="Path to save analysis report JSON")
    
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

    # Default report path
    if not args.output_report:
        args.output_report = os.path.join(os.path.dirname(args.dataset_path), "analysis_results.json")

    manager = DatasetManager(args.dataset_path)
    
    try:
        print(f"Analyzing dataset: {args.dataset_path}")
        results = manager.analyze_dataset(output_report_path=args.output_report)
        print(f"Analysis complete. Report saved to: {args.output_report}")
        
        # Print summary
        rec = results['recommendations']
        print("\nRECOMMENDED PARAMETERS:")
        print(f"  min_area_ratio: {rec['min_area_ratio']:.6f}")
        print(f"  tolerance_ratio: {rec['tolerance_ratio']:.6f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
