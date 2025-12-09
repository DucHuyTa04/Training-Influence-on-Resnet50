"""
Misprediction-Influence Analysis for TracIn

Analyzes mispredictions using TracIn influence scores to identify:
1. Potentially mislabeled training images (high influence but mispredicted)
2. Negative influences (evidence of harmful training data)
3. Error propagation (mislabeled training images causing test errors)

Usage:
    python analyze_misprediction_influences.py
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
    print("="*70)
    print("MISPREDICTION INFLUENCE ANALYSIS")
    print("="*70)
    
    print("\nLoading datasets...")
    train_paths, train_labels, test_paths, test_labels, class_names = load_datasets()
    print(f"Loaded {len(train_paths)} train, {len(test_paths)} test samples")
    
    false_pred_path = parent_dir / 'outputs/mispredictions/false_predictions.csv'
    print(f"Loading mispredictions...")
    false_preds = pd.read_csv(false_pred_path)
    print(f"Found {len(false_preds)} mispredictions")
    
    results_dir = parent_dir / 'outputs/influence_analysis'
    print(f"Loading TracIn results...")
    
    influence_values = np.load(results_dir / 'top_k_influences_values.npy') 
    influence_indices = np.load(results_dir / 'top_k_influences_indices.npy')
    
    num_test, k = influence_values.shape
    print(f"Loaded {num_test} test samples x top-{k} influences")
    
    train_mispred = false_preds[false_preds['split'] == 'train']
    test_mispred = false_preds[false_preds['split'] == 'test']
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(train_mispred)} train, {len(test_mispred)} test mispredictions")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"PART 1: MISPREDICTED TRAINING IMAGES")
    print(f"{'='*70}")
    
    print("Matching mispredicted training images to indices...")
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
    
    print(f"  Matched {len(train_mispred_indices)}/{len(train_mispred)} mispredictions to indices")
    
    # Count appearances of mispredicted training images in top-K
    print(f"\nStep 2: Counting appearances in top-{k} for all test samples...")
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
    
    print(f"  Total appearances: {total_appearances:,}")
    print(f"  Unique mispredicted images appearing: {num_appearing}/{len(train_mispred_indices)}")
    
    if num_appearing > 0:
        avg_appearances = total_appearances / num_appearing
        print(f"  Average appearances per image: {avg_appearances:.1f}")
    
    # Identify most influential mispredicted training images
    print(f"\nStep 3: Ranking mispredicted training images by influence...")
    sorted_appearances = sorted(appearance_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: Top 20 Most Influential Mispredicted Training Images")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Train ID':<10} {'Appearances':<12} {'True→Pred':<20} {'Filename'}")
    print("-" * 100)
    
    for rank, (train_idx, count) in enumerate(sorted_appearances[:20], 1):
        if count == 0:
            break
        info = next(x for x in train_mispred_info if x['train_idx'] == train_idx)
        label_info = f"{info['true_class']}→{info['pred_class']}"
        filename = Path(info['path']).name
        print(f"{rank:<6} {train_idx:<10} {count:<12} {label_info:<20} {filename}")
    
    print(f"\nInterpretation:")
    print(f"  • High appearance count = This mispredicted image influences many test samples")
    print(f"  • These are prime candidates for manual review (potential labeling errors)")
    print(f"  • Notice the pattern in True→Pred (are there common confusions?)")
    
    
    print(f"\n{'='*70}")
    print(f"PART 2: MISPREDICTED TEST IMAGES")
    print(f"{'='*70}")
    print("Hypothesis: If training data is harmful, mispredicted test images")
    print("will show NEGATIVE influences from conflicting training samples.")
    print()
    
    print("Matching mispredicted test images to dataset indices...")
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
    
    print(f"  Matched {len(test_mispred_indices)}/{len(test_mispred)} mispredictions to indices")
    
    # Extract influence scores for all mispredicted test images
    print(f"\nStep 2: Extracting top-{k} influences for mispredicted test images...")
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
    print(f"  Extracted {len(df_mispred_influences):,} influence scores")
    
    # Analyze for negative influences (KEY QUESTION)
    print(f"\nStep 3: Analyzing influence scores for NEGATIVE values...")
    negative_count = (df_mispred_influences['influence'] < 0).sum()
    positive_count = (df_mispred_influences['influence'] >= 0).sum()
    total_count = len(df_mispred_influences)
    
    print(f"\n{'='*70}")
    print(f"CRITICAL FINDING: Influence Score Distribution")
    print(f"{'='*70}")
    print(f"Total influence scores analyzed: {total_count:,}")
    print(f"Positive influences:             {positive_count:,} ({100*positive_count/total_count:.2f}%)")
    print(f"Negative influences:             {negative_count:,} ({100*negative_count/total_count:.2f}%)")
    print()
    
    if negative_count > 0:
        print(f"NEGATIVE INFLUENCES DETECTED!")
        print(f"    This indicates harmful training data that pushes predictions")
        print(f"    in the WRONG direction. These training samples should be reviewed.")
        print()
        print(f"Top 10 Most Negative Influences:")
        most_negative = df_mispred_influences.nsmallest(10, 'influence')
        print(most_negative[['test_idx', 'test_true', 'test_pred', 'train_idx', 'train_label', 'influence', 'rank']].to_string(index=False))
    else:
        print(f"NO NEGATIVE INFLUENCES FOUND")
        print(f"  ")
        print(f"  This suggests:")
        print(f"    • Training labels are generally correct")
        print(f"    • No adversarial/poisoned training data")
        print(f"    • Mispredictions are due to:")
        print(f"      - Ambiguous test cases (boundary examples)")
        print(f"      - Model limitations (insufficient capacity)")
        print(f"      - Natural class overlap (inherently similar)")
        print(f"  ")
        print(f"  Even though influences are positive, highly influential")
        print(f"  mispredicted TRAINING images (from Part 1) may still be")
        print(f"  mislabeled - they teach wrong patterns that happen to align")
        print(f"  with the gradient direction.")
    
    print(f"\nStatistics:")
    print(f"  Min influence:    {df_mispred_influences['influence'].min():.6e}")
    print(f"  Max influence:    {df_mispred_influences['influence'].max():.6e}")
    print(f"  Mean influence:   {df_mispred_influences['influence'].mean():.6e}")
    print(f"  Median influence: {df_mispred_influences['influence'].median():.6e}")
    
    # Save detailed results
    output_path = results_dir / 'misprediction_influence_analysis.csv'
    df_mispred_influences.to_csv(output_path, index=False)
    print(f"\nDetailed analysis saved to: {output_path}")
    
    
    print(f"\n{'='*70}")
    print(f"PART 3: CROSS-ANALYSIS (Error Propagation)")
    print(f"{'='*70}")
    print("Hypothesis: Mislabeled training images will appear as top influences")
    print("for test images with the SAME type of misprediction.")
    print()
    
    print("Finding cross-matches...")
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
                    'same_error': info['pred_class'] == train_info['pred_class']  # Both predicted the same wrong class
                })
    
    print(f"  Found {len(cross_matches):,} cross-matches")
    
    if cross_matches:
        df_cross = pd.DataFrame(cross_matches)
        
        # Count same-error cases
        same_error_count = df_cross['same_error'].sum()
        print(f"  Same error pattern: {same_error_count:,} ({100*same_error_count/len(cross_matches):.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"RESULTS: Evidence of Error Propagation")
        print(f"{'='*70}")
        print(f"When the SAME confusion appears in both train and test mispredictions,")
        print(f"and the training image influences the test image, we have evidence that")
        print(f"the mislabeled training image taught the model an incorrect pattern.")
        print()
        
        # Show top cases
        print(f"Top 10 Cross-Matches (by influence magnitude):")
        top_cross = df_cross.nlargest(10, 'influence')
        
        print(f"\n{'Test':<6} {'TestTrue':<10} {'TestPred':<10} {'Train':<8} {'TrainTrue':<10} {'TrainPred':<10} {'Influence':<12} {'Rank':<6} {'Match?'}")
        print("-" * 100)
        for _, row in top_cross.iterrows():
            match_symbol = "" if row['same_error'] else "---"
            print(f"{row['test_idx']:<6} {row['test_true']:<10} {row['test_pred']:<10} "
                  f"{row['train_idx']:<8} {row['train_true']:<10} {row['train_pred']:<10} "
                  f"{row['influence']:<12.6f} {row['rank']:<6} {match_symbol}")
        
        print()
        print("Legend: = Same error pattern (likely propagation)")
        print("        --- = Different errors (coincidental)")
        
        # Save cross-analysis
        cross_output = results_dir / 'misprediction_cross_analysis.csv'
        df_cross.to_csv(cross_output, index=False)
        print(f"\nCross-analysis saved to: {cross_output}")
    else:
        print(f"\n  No mispredicted training images found in top-{k} for mispredicted test images")
        print(f"  This suggests mispredicted training images don't directly influence test errors.")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    # ===================================================================
    # COMPREHENSIVE CONCLUSIONS
    # ===================================================================
    print(f"\n{'='*70}")
    print("CONCLUSIONS & RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print(f"\n1. NEGATIVE INFLUENCES (Harmful Training Data)")
    print(f"   " + "-" * 65)
    if negative_count > 0:
        print(f"   STATUS: {negative_count:,} NEGATIVE INFLUENCES DETECTED")
        print(f"   ")
        print(f"   ACTION REQUIRED:")
        print(f"   • Review training samples with negative influences")
        print(f"   • Check for mislabeled, corrupted, or adversarial data")
        print(f"   • Remove or relabel problematic training samples")
        print(f"   • Retrain model and verify improvements")
    else:
        print(f"   STATUS: NO NEGATIVE INFLUENCES FOUND")
        print(f"   ")
        print(f"   INTERPRETATION:")
        print(f"   • Training data quality is generally good")
        print(f"   • No evidence of adversarial/poisoned data")
        print(f"   • Mispredictions are due to model limitations or ambiguous cases")
        print(f"   • However, highly influential mispredicted training images")
        print(f"     (from Part 1) may still need manual review")
    
    print(f"\n2. MISPREDICTED TRAINING IMAGES (Potential Label Errors)")
    print(f"   " + "-" * 65)
    if len(train_mispred_indices) > 0:
        most_influential_count = sum(1 for c in appearance_counts.values() if c > 1000)
        print(f"   • Total mispredicted training images: {len(train_mispred_indices)}")
        print(f"   • Images appearing >1000 times in top-{k}: {most_influential_count}")
        print(f"   • Total appearances across all test samples: {total_appearances:,}")
        print(f"   ")
        print(f"   KEY INSIGHT:")
        print(f"   These images are mispredicted YET highly influential.")
        print(f"   This suggests they are defining decision boundaries")
        print(f"   (possibly incorrectly due to mislabeling).")
        print(f"   ")
        print(f"   RECOMMENDED ACTION:")
        print(f"   • Manually inspect top 10-20 mispredicted training images")
        print(f"   • Look for obvious labeling errors (e.g., spiders in butterfly folder)")
        print(f"   • Pay special attention to images with >5000 appearances")
    else:
        print(f"   • No mispredicted training images found (excellent!)")
    
    print(f"\n3. ERROR PROPAGATION (Training → Test)")
    print(f"   " + "-" * 65)
    if cross_matches:
        same_error_pct = 100 * same_error_count / len(cross_matches)
        print(f"   • Cross-matches found: {len(cross_matches):,}")
        print(f"   • Same error pattern: {same_error_count:,} ({same_error_pct:.1f}%)")
        print(f"   ")
        if same_error_pct > 50:
            print(f"   HIGH ERROR PROPAGATION DETECTED!")
            print(f"   More than half of cross-matches show the same error pattern.")
            print(f"   This strongly suggests mislabeled training data is teaching")
            print(f"   the model incorrect patterns that propagate to test errors.")
        else:
            print(f"   Moderate error propagation")
            print(f"   Some training mispredictions influence test errors, but")
            print(f"   the pattern is not consistent across all cases.")
    else:
        print(f"   • No error propagation detected")
        print(f"   • Mispredicted training images don't appear in top-{k} for test errors")
    
    print(f"\n4. OVERALL ASSESSMENT")
    print(f"   " + "-" * 65)
    
    # Calculate overall metrics
    train_mispred_rate = len(train_mispred) / len(train_paths) * 100 if len(train_paths) > 0 else 0
    test_mispred_rate = len(test_mispred) / len(test_paths) * 100 if len(test_paths) > 0 else 0
    
    print(f"   Model Performance:")
    print(f"   • Training error rate: {train_mispred_rate:.2f}% ({len(train_mispred)}/{len(train_paths)})")
    print(f"   • Test error rate:     {test_mispred_rate:.2f}% ({len(test_mispred)}/{len(test_paths)})")
    print(f"   ")
    
    if negative_count == 0 and most_influential_count > 0:
        print(f"   DIAGNOSIS: Clean dataset with some boundary cases")
        print(f"   • No harmful training data detected")
        print(f"   • {most_influential_count} highly influential mispredicted images need review")
        print(f"   • Model quality can likely improve with targeted data cleaning")
        print(f"   ")
        print(f"   NEXT STEPS:")
        print(f"   1. Manually inspect top 20 mispredicted training images")
        print(f"   2. Fix any obvious labeling errors")
        print(f"   3. Retrain and measure improvement")
    elif negative_count > 0:
        print(f"   DIAGNOSIS: Harmful training data detected")
        print(f"   • {negative_count:,} negative influences require immediate attention")
        print(f"   • Training data quality issues need resolution")
        print(f"   ")
        print(f"   NEXT STEPS:")
        print(f"   1. Identify and review all training samples with negative influence")
        print(f"   2. Remove or relabel problematic samples")
        print(f"   3. Retrain model on cleaned dataset")
        print(f"   4. Re-run this analysis to verify improvements")
    else:
        print(f"   DIAGNOSIS: Excellent data quality")
        print(f"   • No mispredicted training images")
        print(f"   • No negative influences")
        print(f"   • Test errors are likely due to inherently difficult cases")
    
    print(f"\n{'='*70}")
    print(f"FILES GENERATED:")
    print(f"{'='*70}")
    print(f"  • misprediction_influence_analysis.csv - Detailed influence scores")
    if cross_matches:
        print(f"  • misprediction_cross_analysis.csv - Error propagation analysis")
    print(f"\n{'='*70}")

if __name__ == '__main__':
    main()
