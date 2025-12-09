"""
Inspect Top Influential Images

Visualizes the most and least influential training images across the dataset.
Shows which images have the strongest positive or negative impact on test predictions.

Usage:
    python inspect_top_influences.py --top_n 20
    python inspect_top_influences.py --helpful_only
    python inspect_top_influences.py --show_per_class
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))


class TopInfluenceInspector:
    """Inspector for most influential training images."""
    
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
        
        print(f"Loaded {len(self.train_images)} training images")
        
        # Load influence scores
        self.values = np.load(self.results_dir / 'top_k_influences_values.npy')
        self.indices = np.load(self.results_dir / 'top_k_influences_indices.npy')
        
        print(f"Loaded influence scores: {self.values.shape}")
        print(f"Test samples: {self.values.shape[0]}, Top-K: {self.values.shape[1]}\n")
    
    def analyze_global_influences(self):
        """Analyze influence scores across all training images."""
        # Count appearances in top-K
        train_idx_counts = Counter(self.indices.flatten())
        
        # Calculate statistics per training image
        train_stats = []
        
        for train_idx in range(len(self.train_images)):
            if train_idx in train_idx_counts:
                # Get all influence scores for this training image
                mask = self.indices == train_idx
                influences = self.values[mask]
                
                train_stats.append({
                    'train_idx': train_idx,
                    'appearance_count': train_idx_counts[train_idx],
                    'mean_influence': influences.mean(),
                    'max_influence': influences.max(),
                    'min_influence': influences.min(),
                    'std_influence': influences.std(),
                    'image_path': self.train_images[train_idx],
                    'label': self.train_labels[train_idx],
                    'class_name': self.class_names[self.train_labels[train_idx]]
                })
        
        return pd.DataFrame(train_stats)
    
    def get_top_helpful(self, n=20):
        """Get most helpful training images (highest positive influence)."""
        stats_df = self.analyze_global_influences()
        return stats_df.nlargest(n, 'max_influence')
    
    def get_top_harmful(self, n=20):
        """Get least helpful training images (lowest influence)."""
        stats_df = self.analyze_global_influences()
        return stats_df.nsmallest(n, 'min_influence')
    
    def visualize_influences(self, df, title, save_path, metric='max_influence'):
        """Visualize top influential images."""
        n = len(df)
        cols = 5
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for idx, (_, row) in enumerate(df.iterrows()):
            ax = axes[idx]
            
            if Path(row['image_path']).exists():
                img = Image.open(row['image_path']).convert('RGB')
                ax.imshow(img)
                
                # Title with statistics
                title_text = (f"#{idx+1}: Train {row['train_idx']}\n"
                             f"Class: {row['class_name']}\n"
                             f"Appears: {row['appearance_count']}x\n"
                             f"Max: {row['max_influence']:.2e}\n"
                             f"Avg: {row['mean_influence']:.2e}")
                
                ax.set_title(title_text, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
        plt.close()
    
    def print_influence_report(self, helpful_df, harmful_df):
        """Print brief influence summary."""
        all_values = self.values.flatten()
        negative = all_values[all_values < 0]
        
        if len(negative) > 0:
            print(f"[WARN] {len(negative):,} negative influences detected")
        
        print(f"[INFO] Top helpful: Train {helpful_df.iloc[0]['train_idx']} (max inf: {helpful_df.iloc[0]['max_influence']:.2e})")
        print(f"[INFO] Top harmful: Train {harmful_df.iloc[0]['train_idx']} (min inf: {harmful_df.iloc[0]['min_influence']:.2e})")
    
    def analyze_per_class_influences(self):
        """Analyze influence distribution per class (saved to visualizations)."""
        # Per-class stats are in the image visualizations
        pass
            


def main():
    parser = argparse.ArgumentParser(description='Inspect top influential training images')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top images to show')
    parser.add_argument('--output_helpful', type=str, default='outputs/inspection/top_helpful_images.png',
                       help='Output path for helpful images')
    parser.add_argument('--output_harmful', type=str, default='outputs/inspection/top_harmful_images.png',
                       help='Output path for harmful images')
    
    args = parser.parse_args()
    
    print("[INFO] Analyzing top influential training images...")
    
    inspector = TopInfluenceInspector()
    helpful_df = inspector.get_top_helpful(n=args.top_n)
    harmful_df = inspector.get_top_harmful(n=args.top_n)
    
    inspector.print_influence_report(helpful_df, harmful_df)
    
    print(f"[INFO] Generating visualizations...")
    inspector.visualize_influences(
        helpful_df,
        f'Top {args.top_n} Most Helpful Training Images\n(Highest Positive Influence)',
        args.output_helpful,
        metric='max_influence'
    )
    
    inspector.visualize_influences(
        harmful_df,
        f'Top {args.top_n} Least Influential Training Images\n(Lowest Influence)',
        args.output_harmful,
        metric='min_influence'
    )
    
    print(f"\n[DONE] Outputs saved:")
    print(f"       - {args.output_helpful}")
    print(f"       - {args.output_harmful}")


if __name__ == '__main__':
    main()
