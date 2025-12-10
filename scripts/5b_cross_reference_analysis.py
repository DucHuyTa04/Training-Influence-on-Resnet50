"""
Analyzes mispredictions to identify:
1. Potentially mislabeled training images (high influence but mispredicted)
2. Negative influences (evidence of harmful training data)
3. Error propagation (mislabeled training images causing test errors)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

def load_datasets():
    """
    Load training and test datasets to get file paths and labels.
    
    Returns:
        train_paths: List of Path objects for training images
        train_labels: List of integer labels for training images
        test_paths: List of Path objects for test images
        test_labels: List of integer labels for test images
        class_names: List of class names (e.g., ['butterfly', 'cat', ...])
    """
    data_dir = parent_dir / 'data' / 'processed'
    
    train_dataset = datasets.ImageFolder(root=str(data_dir / 'train'))
    test_dataset = datasets.ImageFolder(root=str(data_dir / 'test'))
    
    # Extract file paths and labels from the dataset
    train_paths = [Path(path) for path, _ in train_dataset.samples]
    train_labels = [label for _, label in train_dataset.samples]
    
    test_paths = [Path(path) for path, _ in test_dataset.samples]
    test_labels = [label for _, label in test_dataset.samples]
    
    return train_paths, train_labels, test_paths, test_labels, train_dataset.classes

def match_path_to_index(target_path, dataset_paths):
    """
    Match a file path to its index in the dataset.
    
    Strategy:
    1. Try exact path match first (most reliable)
    2. Fall back to filename-only match if exact match fails
    3. Warn if multiple matches found (ambiguous)
    
    Args:
        target_path: Path to the file we're trying to match
        dataset_paths: List of Path objects from the dataset
        
    Returns:
        Index in the dataset, or None if no match found
    """
    target_path = Path(target_path)
    
    # Strategy 1: Exact path match
    for idx, path in enumerate(dataset_paths):
        if path == target_path:
            return idx
    
    # Strategy 2: Match by filename only (in case of path differences)
    target_name = target_path.name
    matches = []
    for idx, path in enumerate(dataset_paths):
        if path.name == target_name:
            matches.append(idx)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple files with same name - return first match but warn
        print(f"  Warning: Multiple matches for {target_name}: {matches}, using first")
        return matches[0]
    
    # No match found
    return None

def main():
    print("[INFO] Misprediction-Influence Cross-Analysis")
    
    # Load data
    train_paths, train_labels, test_paths, test_labels, class_names = load_datasets()
    
    false_pred_path = parent_dir / 'outputs/mispredictions/false_predictions.csv'
    false_preds = pd.read_csv(false_pred_path)
    
    results_dir = parent_dir / 'outputs/influence_analysis'
    influence_values = np.load(results_dir / 'top_k_influences_values.npy') 
    influence_indices = np.load(results_dir / 'top_k_influences_indices.npy')
    
    num_test, k = influence_values.shape
    
    train_mispred = false_preds[false_preds['split'] == 'train']
    test_mispred = false_preds[false_preds['split'] == 'test']
    
    print(f"[INFO] Found {len(train_mispred)} train mispredictions, {len(test_mispred)} test mispredictions")
    
    # Part 1: Analyze mispredicted training images
    train_mispred_indices = []
    train_mispred_info = []
    
    for idx, row in train_mispred.iterrows():
        img_path = row['image_path']
        train_idx = match_path_to_index(img_path, train_paths)
        
        if train_idx is not None:
            train_mispred_indices.append(train_idx)
            train_mispred_info.append({
                'train_idx': train_idx,
                'path': img_path,
                'true_label': row['true_class_index'],
                'pred_label': row['pred_class_index'],
                'true_class': class_names[row['true_class_index']],
                'pred_class': class_names[row['pred_class_index']]
            })
    
    # Count appearances in top-K
    train_mispred_set = set(train_mispred_indices)
    appearance_counts = {idx: 0 for idx in train_mispred_indices}
    total_appearances = 0
    
    for test_idx in range(num_test):
        top_k_train = influence_indices[test_idx]
        for train_idx in top_k_train:
            if train_idx in train_mispred_set:
                appearance_counts[train_idx] += 1
                total_appearances += 1
    
    num_appearing = sum(1 for c in appearance_counts.values() if c > 0)
    sorted_appearances = sorted(appearance_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Show top 5 most influential mispredicted images
    high_priority = [train_idx for train_idx, count in sorted_appearances[:20] if count > 1000]
    if len(high_priority) > 0:
        print(f"[WARN] {len(high_priority)} mispredicted training images with >1000 appearances (likely mislabeled)")
        print(f"       Top candidates: {high_priority[:5]}")
    test_mispred_indices = []
    test_mispred_info = []
    
    for idx, row in test_mispred.iterrows():
        img_path = row['image_path']
        test_idx = match_path_to_index(img_path, test_paths)
        
        if test_idx is not None:
            test_mispred_indices.append(test_idx)
            test_mispred_info.append({
                'test_idx': test_idx,
                'path': img_path,
                'true_label': row['true_class_index'],
                'pred_label': row['pred_class_index'],
                'true_class': class_names[row['true_class_index']],
                'pred_class': class_names[row['pred_class_index']]
            })
    
    # Extract influence scores for mispredicted test images
    mispred_test_influences = []
    
    for info in test_mispred_info:
        test_idx = info['test_idx']
        top_k_values = influence_values[test_idx]
        top_k_train = influence_indices[test_idx]
        
        for rank, (train_idx, influence) in enumerate(zip(top_k_train, top_k_values), 1):
            mispred_test_influences.append({
                'test_idx': test_idx,
                'test_true': info['true_class'],
                'test_pred': info['pred_class'],
                'train_idx': int(train_idx),
                'train_label': class_names[train_labels[int(train_idx)]],
                'influence': float(influence),
                'rank': rank
            })
    
    df_mispred_influences = pd.DataFrame(mispred_test_influences)
    
    # Check for negative influences
    negative_count = (df_mispred_influences['influence'] < 0).sum()
    total_count = len(df_mispred_influences)
    
    if negative_count > 0:
        print(f"[WARN] {negative_count:,} negative influences detected ({100*negative_count/total_count:.1f}%)")
        print(f"       This indicates potentially harmful training data")
    else:
        print(f"[INFO] No negative influences found - training data appears clean")
    
    # Save analysis
    output_path = results_dir / 'misprediction_influence_analysis.csv'
    df_mispred_influences.to_csv(output_path, index=False)
    
    # Part 3: Cross-analysis (error propagation)
    cross_matches = []
    for info in test_mispred_info:
        test_idx = info['test_idx']
        top_k_train = influence_indices[test_idx]
        
        for rank, train_idx in enumerate(top_k_train, 1):
            if train_idx in train_mispred_set:
                train_info = next(x for x in train_mispred_info if x['train_idx'] == train_idx)
                influence = influence_values[test_idx][rank - 1]
                
                cross_matches.append({
                    'test_idx': test_idx,
                    'test_true': info['true_class'],
                    'test_pred': info['pred_class'],
                    'train_idx': int(train_idx),
                    'train_true': train_info['true_class'],
                    'train_pred': train_info['pred_class'],
                    'influence': float(influence),
                    'rank': rank,
                    'same_error': info['pred_class'] == train_info['pred_class']
                })
    
    if cross_matches:
        df_cross = pd.DataFrame(cross_matches)
        same_error_count = df_cross['same_error'].sum()
        same_error_pct = 100 * same_error_count / len(cross_matches)
        
        if same_error_pct > 50:
            print(f"[WARN] High error propagation detected ({same_error_pct:.1f}%)")
            print(f"       Same error pattern in {same_error_count}/{len(cross_matches)} cross-matches")
        
        cross_output = results_dir / 'misprediction_cross_analysis.csv'
        df_cross.to_csv(cross_output, index=False)
    
    print(f"\n[DONE] Analysis complete")
    print(f"       Results saved to:")
    print(f"       - misprediction_influence_analysis.csv")
    if cross_matches:
        print(f"       - misprediction_cross_analysis.csv")

if __name__ == '__main__':
    main()
