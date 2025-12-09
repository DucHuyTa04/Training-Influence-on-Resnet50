#!/usr/bin/env python3
"""
Temporary script to update all file paths in the reorganized scripts.
Run once after restructuring, then delete this file.
"""

import re
from pathlib import Path

# Path mappings (old -> new)
PATH_MAPPINGS = {
    # Model paths
    "models/weights_cache": "models/pretrained",
    "models/Resnet50": "models/best/Resnet50",
    "'models/checkpoints'": "str(script_dir / 'models' / 'checkpoints')",
    
    # Data paths
    "data/raw-img": "data/raw",
    
    # Output paths
    "false_predictions.csv": "outputs/mispredictions/false_predictions.csv",
    "false_predictions/combined_mispredictions.png": "outputs/mispredictions/mispredictions_grid.png",
    "mislabeled_inspection.png": "outputs/inspection/mislabeled_candidates.png",
    "top_helpful_inspection.png": "outputs/inspection/top_helpful_images.png",
    "top_harmful_inspection.png": "outputs/inspection/top_harmful_images.png",
    
    # TracIn outputs
    "TracIn/results/top_k_influences_values.npy": "outputs/influence_analysis/influence_scores.npy",
    "TracIn/results/top_k_influences_indices.npy": "outputs/influence_analysis/influence_indices.npy",
    "TracIn/results/top_k_influences.csv": "outputs/influence_analysis/influence_scores.csv",
    "TracIn/results/overall_dashboard.png": "outputs/influence_analysis/overall_dashboard.png",
    "TracIn/results/per_class": "outputs/influence_analysis/per_class",
    "TracIn/results/misprediction_influence_analysis.csv": "outputs/influence_analysis/misprediction_influence_analysis.csv",
    "TracIn/results/misprediction_cross_analysis.csv": "outputs/influence_analysis/misprediction_cross_analysis.csv",
    "TracIn/top_influential/helpful": "outputs/inspection/detailed/helpful",
    "TracIn/top_influential/harmful": "outputs/inspection/detailed/harmful",
}

def update_file(filepath):
    """Update paths in a single file."""
    print(f"Updating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    for old_path, new_path in PATH_MAPPINGS.items():
        content = content.replace(old_path, new_path)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [UPDATED] {filepath}")
        return True
    else:
        print(f"  - No changes needed")
        return False

def main():
    script_dir = Path(__file__).parent
    scripts_path = script_dir / 'scripts'
    
    python_files = list(scripts_path.glob('*.py')) + list(scripts_path.glob('utils/*.py'))
    
    updated_count = 0
    for filepath in sorted(python_files):
        if update_file(filepath):
            updated_count += 1
    
    print(f"\n[DONE] Updated {updated_count} files")
    print("\nYou can now delete this script: rm update_paths.py")

if __name__ == '__main__':
    main()
