"""
Inspect Mislabeled Images

Analyzes the most likely mislabeled training images based on TracIn influence scores
and misprediction patterns. Shows visual grids and detailed statistics.
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
        
        # Load datasets
        self.train_images = []
        self.train_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = train_dir / class_name
            images = sorted([f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            self.train_images.extend(images)
            self.train_labels.extend([class_idx] * len(images))
        
        # Load mispredictions (version-aware path)
        false_pred_path = self.results_dir.parent / 'mispredictions' / 'false_predictions.csv'
        self.false_preds = pd.read_csv(false_pred_path)
        
        # Load influence cross-analysis
        cross_analysis_path = self.results_dir / 'misprediction_cross_analysis.csv'
        self.cross_analysis = pd.read_csv(cross_analysis_path)
    
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
            path = Path(row['path'])
            for idx, img_path in enumerate(self.train_images):
                if img_path.name == path.name:
                    pred_map[idx] = row['pred']
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
        """Print brief summary of mislabeled candidates."""
        high_priority = candidates_df[candidates_df['appearance_count'] > 1000]
        
        if len(high_priority) > 0:
            print(f"\n[WARN] {len(high_priority)} high-priority mislabeled candidates (>1000 appearances)")
            for idx, row in high_priority.head(5).iterrows():
                print(f"       Train {row['train_idx']}: {row['true_class']}→{row['pred_class']} ({row['appearance_count']} appearances)")
        
        # Just print top 5 if not many high priority ones
        if len(high_priority) < 5:
            print(f"\n[INFO] Top {min(5, len(candidates_df))} mislabeled candidates:")
            for idx, row in candidates_df.head(5).iterrows():
                print(f"       Train {row['train_idx']}: {row['true_class']}→{row['pred_class']} ({row['appearance_count']} appearances)")
    
    def generate_action_items(self, candidates_df, threshold=1000):
        """Save action items to file."""
        # Actions are in the visual grid and detailed CSV file
        pass


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'utils'))
    from version_manager import VersionManager
    
    parser = argparse.ArgumentParser(description='Inspect potentially mislabeled training images')
    parser.add_argument('--version', type=int, help='Model version number')
    parser.add_argument('--results_dir', type=str, help='Results directory (overrides version)')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top candidates to inspect')
    parser.add_argument('--threshold', type=int, default=1000, help='High-priority threshold')
    parser.add_argument('--output', type=str, help='Output image path (overrides version)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    
    if args.results_dir:
        results_dir = args.results_dir
        output_path = args.output if args.output else 'outputs/inspection/mislabeled_candidates.png'
    elif args.version:
        results_dir = f'outputs/v{args.version}/influence_analysis'
        output_path = args.output if args.output else f'outputs/v{args.version}/inspection/mislabeled_candidates.png'
    else:
        vm = VersionManager(script_dir)
        latest = vm.get_latest_version()
        if latest is None:
            print("[ERROR] No trained models found. Use --version or train a model first.")
            return
        results_dir = f'outputs/v{latest}/influence_analysis'
        output_path = args.output if args.output else f'outputs/v{latest}/inspection/mislabeled_candidates.png'
        print(f"[VERSION] Using latest model: v{latest}")
    
    print(f"[INFO] Analyzing mislabeled candidates")
    
    inspector = MislabeledInspector(results_dir=results_dir)
    candidates = inspector.find_top_mislabeled_candidates(top_n=args.top_n)
    
    inspector.print_detailed_report(candidates)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    inspector.visualize_mislabeled_grid(candidates, save_path=output_path)
    
    report_dir = Path(output_path).parent
    report_path = report_dir / 'mislabeled_inspection_report.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(report_path, index=False)
    
    print(f"[DONE] Saved to {output_path}")


if __name__ == '__main__':
    main()
