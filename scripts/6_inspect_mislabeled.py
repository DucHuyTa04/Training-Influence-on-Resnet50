"""
Inspect Mislabeled Images

Analyzes the most likely mislabeled training images based on TracIn influence scores
and misprediction patterns. Shows visual grids and detailed statistics.

Usage:
    python inspect_mislabeled.py --top_n 20
    python inspect_mislabeled.py --top_n 50 --save_report
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))


class MislabeledInspector:
    """Inspector for potentially mislabeled training images."""
    
    def __init__(self, data_dir='data/processed', results_dir='outputs/influence_analysis'):
        script_dir = Path(__file__).parent.parent
        self.data_dir = script_dir / data_dir
        self.results_dir = script_dir / results_dir
        
        # Load class names
        train_dir = self.data_dir / 'train'
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"Classes: {', '.join(self.class_names)}")
        
        # Load datasets
        self.train_images = []
        self.train_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = train_dir / class_name
            images = sorted([f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            self.train_images.extend(images)
            self.train_labels.extend([class_idx] * len(images))
        
        print(f"Loaded {len(self.train_images)} training images")
        
        # Load mispredictions
        false_pred_path = 'outputs/mispredictions/false_predictions.csv'
        self.false_preds = pd.read_csv(false_pred_path)
        print(f"Loaded {len(self.false_preds)} mispredictions")
        
        # Load influence cross-analysis
        cross_analysis_path = self.results_dir / 'misprediction_cross_analysis.csv'
        self.cross_analysis = pd.read_csv(cross_analysis_path)
        print(f"Loaded {len(self.cross_analysis)} cross-analysis records\n")
    
    def find_top_mislabeled_candidates(self, top_n=20):
        """
        Find training images most likely to be mislabeled.
        
        Criteria:
        1. Training image is mispredicted
        2. Appears frequently in top influences (high impact)
        3. Shows same error pattern across multiple test samples
        
        Returns:
            DataFrame with top mislabeled candidates
        """
        # Filter for training mispredictions with same error pattern
        same_error = self.cross_analysis[self.cross_analysis['same_error'] == True].copy()
        
        # Count appearances per training image
        train_appearances = same_error.groupby('train_idx').agg({
            'influence': ['count', 'sum', 'mean', 'max'],
            'test_idx': 'nunique'
        }).reset_index()
        
        train_appearances.columns = ['train_idx', 'appearance_count', 'total_influence', 
                                     'mean_influence', 'max_influence', 'unique_tests']
        
        # Sort by appearance count (most suspicious)
        train_appearances = train_appearances.sort_values('appearance_count', ascending=False)
        
        # Get top N
        top_candidates = train_appearances.head(top_n).copy()
        
        # Add image details
        top_candidates['image_path'] = top_candidates['train_idx'].apply(
            lambda idx: self.train_images[idx] if idx < len(self.train_images) else None
        )
        top_candidates['true_label'] = top_candidates['train_idx'].apply(
            lambda idx: self.train_labels[idx] if idx < len(self.train_labels) else -1
        )
        top_candidates['true_class'] = top_candidates['true_label'].apply(
            lambda idx: self.class_names[idx] if 0 <= idx < len(self.class_names) else 'unknown'
        )
        
        # Get predicted label from false predictions
        train_false_preds = self.false_preds[self.false_preds['split'] == 'train'].copy()
        
        pred_map = {}
        for _, row in train_false_preds.iterrows():
            # Match by filename
            path = Path(row['image_path'])
            for idx, img_path in enumerate(self.train_images):
                if img_path.name == path.name:
                    pred_map[idx] = row['pred_class_index']
                    break
        
        top_candidates['pred_label'] = top_candidates['train_idx'].apply(
            lambda idx: pred_map.get(idx, -1)
        )
        top_candidates['pred_class'] = top_candidates['pred_label'].apply(
            lambda idx: self.class_names[idx] if 0 <= idx < len(self.class_names) else 'unknown'
        )
        
        return top_candidates
    
    def visualize_mislabeled_grid(self, candidates_df, save_path='outputs/inspection/mislabeled_candidates.png'):
        """Create visual grid of top mislabeled candidates."""
        n = len(candidates_df)
        cols = 5
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for idx, (_, row) in enumerate(candidates_df.iterrows()):
            ax = axes[idx]
            
            if row['image_path'] and Path(row['image_path']).exists():
                img = Image.open(row['image_path']).convert('RGB')
                ax.imshow(img)
                
                # Title with error pattern
                title = (f"#{idx+1}: Train {row['train_idx']}\n"
                        f"Label: {row['true_class']} → {row['pred_class']}\n"
                        f"Appears: {row['appearance_count']}x (in {row['unique_tests']} tests)\n"
                        f"Max Inf: {row['max_influence']:.2e}")
                
                # Color based on severity
                if row['appearance_count'] > 1000:
                    color = 'red'
                elif row['appearance_count'] > 500:
                    color = 'green'
                else:
                    color = 'black'
                
                ax.set_title(title, fontsize=9, color=color, weight='bold')
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Top Mislabeled Training Image Candidates\n(Sorted by Appearance Count in Top Influences)', 
                     fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
        plt.close()
    
    def print_detailed_report(self, candidates_df):
        """Print detailed report of mislabeled candidates."""
        print("\n" + "="*80)
        print("MISLABELED IMAGE CANDIDATES - DETAILED REPORT")
        print("="*80)
        
        total_train = len(self.train_images)
        total_test = len([f for f in self.false_preds if f != 'train'])
        
        print(f"\nDataset Summary:")
        print(f"  Total training images: {total_train:,}")
        print(f"  Training mispredictions: {len(self.false_preds[self.false_preds['split'] == 'train']):,}")
        print(f"  Test mispredictions: {len(self.false_preds[self.false_preds['split'] == 'test']):,}")
        
        print(f"\nTop {len(candidates_df)} Mislabeled Candidates:")
        print("-" * 80)
        
        for idx, row in candidates_df.iterrows():
            print(f"\n#{idx+1}. Train Index {row['train_idx']}: {row['image_path'].name if row['image_path'] else 'N/A'}")
            print(f"   Label: {row['true_class']} ({row['true_label']}) → Predicted: {row['pred_class']} ({row['pred_label']})")
            print(f"   Appears in top-100 influences: {row['appearance_count']} times")
            print(f"   Affects {row['unique_tests']} unique test samples")
            print(f"   Mean influence: {row['mean_influence']:.6e}")
            print(f"   Max influence: {row['max_influence']:.6e}")
            
            # Percentage of test set
            pct = (row['appearance_count'] / 5224) * 100  # Assuming 5224 test samples
            print(f"   Impact: {pct:.1f}% of test set")
            
            if pct > 100:
                print(f"   ⚠️  WARNING: Appears more times than test samples exist!")
                print(f"              This image is HIGHLY LIKELY to be mislabeled")
        
        print("\n" + "="*80)
        
        # Error pattern analysis
        print("\nError Pattern Analysis:")
        print("-" * 80)
        
        error_patterns = candidates_df.groupby(['true_class', 'pred_class']).size().sort_values(ascending=False)
        for (true_cls, pred_cls), count in error_patterns.head(10).items():
            print(f"  {true_cls} → {pred_cls}: {count} images")
    
    def generate_action_items(self, candidates_df, threshold=1000):
        """Generate actionable recommendations."""
        high_priority = candidates_df[candidates_df['appearance_count'] > threshold]
        
        print("\n" + "="*80)
        print("RECOMMENDED ACTIONS")
        print("="*80)
        
        if len(high_priority) > 0:
            print(f"\n🔴 HIGH PRIORITY: {len(high_priority)} images with >{threshold} appearances")
            print("\nThese images should be manually reviewed IMMEDIATELY:")
            
            for idx, row in high_priority.iterrows():
                print(f"  • {row['image_path'].name}: {row['true_class']} → {row['pred_class']} "
                      f"({row['appearance_count']} appearances)")
        
        medium_priority = candidates_df[
            (candidates_df['appearance_count'] > 500) & 
            (candidates_df['appearance_count'] <= threshold)
        ]
        
        if len(medium_priority) > 0:
            print(f"\n🟡 MEDIUM PRIORITY: {len(medium_priority)} images with 500-{threshold} appearances")
            print("  Review when time permits")
        
        print(f"\n📋 Next Steps:")
        print(f"  1. Open 'outputs/inspection/mislabeled_candidates.png' to visually inspect flagged images")
        print(f"  2. Check original dataset source for these specific images")
        print(f"  3. If mislabeled, move to correct folder or remove")
        print(f"  4. Document changes in a log file")
        print(f"  5. Retrain model: python scripts/2_train.py")
        print(f"  6. Rerun TracIn: python scripts/4_compute_influence.py --top_k 100")
        print(f"  7. Compare new outputs/mispredictions/false_predictions.csv with current version")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect potentially mislabeled training images')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top candidates to inspect')
    parser.add_argument('--save_report', action='store_true', help='Save detailed report to file')
    parser.add_argument('--threshold', type=int, default=1000, help='High-priority threshold')
    parser.add_argument('--output', type=str, default='outputs/inspection/mislabeled_candidates.png', 
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MISLABELED IMAGE INSPECTOR")
    print("="*80 + "\n")
    
    # Initialize inspector
    inspector = MislabeledInspector()
    
    # Find candidates
    print(f"Analyzing top {args.top_n} mislabeled candidates...\n")
    candidates = inspector.find_top_mislabeled_candidates(top_n=args.top_n)
    
    # Print report
    inspector.print_detailed_report(candidates)
    
    # Generate action items
    inspector.generate_action_items(candidates, threshold=args.threshold)
    
    # Visualize
    inspector.visualize_mislabeled_grid(candidates, save_path=args.output)
    
    # Save report if requested
    if args.save_report:
        report_path = 'mislabeled_inspection_report.csv'
        candidates.to_csv(report_path, index=False)
        print(f"Saved detailed report to: {report_path}")


if __name__ == '__main__':
    main()
