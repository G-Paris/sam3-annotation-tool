#!/usr/bin/env python3
import os
import sys
import zipfile
import argparse

def remove_zone_identifiers(dataset_path):
    """Remove Windows 'Zone.Identifier' files that may have been copied over."""
    print("Checking for Zone.Identifier files...")
    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(":Zone.Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
    
    if count > 0:
        print(f"Removed {count} 'Zone.Identifier' files.")
    else:
        print("No 'Zone.Identifier' files found.")
    print()

def create_empty_validation(dataset_path):
    """Create empty validation folders to prevent CVAT duplication."""
    print("Ensuring empty validation folders exist...")
    val_images = os.path.join(dataset_path, "images", "val")
    val_labels = os.path.join(dataset_path, "labels", "val")
    
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    
    # Update data.yaml if needed
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('val:'):
                continue # Remove val line
            else:
                new_lines.append(line)
        
        with open(yaml_path, 'w') as f:
            f.writelines(new_lines)
            
    print("Validation folders ready (but not added to data.yaml).")

def zip_dataset(dataset_path, output_path):
    """Zip the dataset (images, labels, data.yaml) excluding junk."""
    print(f"Zipping dataset to {output_path}...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add data.yaml
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            zipf.write(yaml_path, "data.yaml")
            
        # Add images and labels
        for subdir in ["images", "labels"]:
            dir_path = os.path.join(dataset_path, subdir)
            if not os.path.exists(dir_path):
                continue
                
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(":Zone.Identifier"):
                        continue
                    if file.startswith("."): # Skip hidden files like .DS_Store
                        continue
                        
                    file_path = os.path.join(root, file)
                    # Calculate relative path for zip
                    rel_path = os.path.relpath(file_path, dataset_path)
                    zipf.write(file_path, rel_path)
                    
    print(f"Dataset zipped successfully: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "output/GOB_SAM3_dataset"

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist")
        sys.exit(1)

    remove_zone_identifiers(dataset_path)
    create_empty_validation(dataset_path)
    
    zip_name = f"{os.path.basename(dataset_path)}.zip"
    zip_path = os.path.join(os.path.dirname(dataset_path), zip_name)
    zip_dataset(dataset_path, zip_path)
