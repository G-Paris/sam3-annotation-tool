#!/usr/bin/env python3
"""
Clean up YOLO dataset polygons.
"""

import os
import sys
import argparse
import json

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sam3_annotation_tool.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="Clean up YOLO dataset polygons")
    parser.add_argument("dataset_path", nargs='?', help="Path to dataset root")
    parser.add_argument("--tolerance-ratio", type=float, help="Simplification tolerance ratio")
    parser.add_argument("--min-area-ratio", type=float, help="Minimum area ratio")
    parser.add_argument("--output-report", help="Path to save cleanup report JSON")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes")
    parser.add_argument("--skip-preview", action="store_true", help="Skip preview (deprecated, always skips now)")
    
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
        args.output_report = os.path.join(os.path.dirname(args.dataset_path), "cleanup_report.json")

    manager = DatasetManager(args.dataset_path)
    
    # Load analysis results if parameters not provided
    if args.tolerance_ratio is None or args.min_area_ratio is None:
        analysis_file = os.path.join(os.path.dirname(args.dataset_path), "analysis_results.json")
        if os.path.exists(analysis_file):
            print(f"Loading parameters from {analysis_file}")
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            args.tolerance_ratio = args.tolerance_ratio or analysis['recommendations']['tolerance_ratio']
            args.min_area_ratio = args.min_area_ratio or analysis['recommendations']['min_area_ratio']
        else:
            print("ERROR: No analysis results found and no parameters provided.")
            sys.exit(1)

    try:
        # Remove Zone.Identifier files
        manager.remove_zone_identifiers()
        
        print(f"Cleaning dataset: {args.dataset_path}")
        print(f"  tolerance_ratio: {args.tolerance_ratio:.6f}")
        print(f"  min_area_ratio: {args.min_area_ratio:.6f}")
        
        stats = manager.cleanup_dataset(
            args.tolerance_ratio, 
            args.min_area_ratio, 
            output_report_path=args.output_report,
            dry_run=args.dry_run
        )
        
        print(f"Cleanup complete. Report saved to: {args.output_report}")
        print(f"  Polygons removed: {stats['total_filtered']}")
        print(f"  Points reduced: {stats['total_points_before']} -> {stats['total_points_after']}")
        
        if not args.dry_run:
            print("\nFinalizing dataset...")
            zip_path = manager.finalize_dataset(create_zip=True)
            print(f"Dataset zipped to: {zip_path}")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
