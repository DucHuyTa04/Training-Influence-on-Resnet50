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
        """Print detailed influence report."""
        print("\n" + "="*80)
        print("TOP INFLUENTIAL IMAGES REPORT")
        print("="*80)
        
        # Overall statistics
        all_values = self.values.flatten()
        print(f"\nOverall Statistics:")
        print(f"  Total influence scores: {len(all_values):,}")
        print(f"  Mean influence: {all_values.mean():.6e}")
        print(f"  Std influence: {all_values.std():.6e}")
        print(f"  Max influence: {all_values.max():.6e}")
        print(f"  Min influence: {all_values.min():.6e}")
        
        # Positive vs negative
        positive = all_values[all_values >= 0]
        negative = all_values[all_values < 0]
        
        print(f"\nInfluence Distribution:")
        print(f"  Positive influences: {len(positive):,} ({100*len(positive)/len(all_values):.2f}%)")
        print(f"  Negative influences: {len(negative):,} ({100*len(negative)/len(all_values):.2f}%)")
        
        if len(negative) > 0:
            print(f"  ⚠️  WARNING: Negative influences detected (potential harmful data)")
        else:
            print(f"  ✓ No negative influences (clean dataset)")
        
        # Top helpful images
        print("\n" + "-"*80)
        print("TOP HELPFUL IMAGES (Highest Positive Influence)")
        print("-"*80)
        
        for idx, row in helpful_df.head(10).iterrows():
            print(f"\n#{idx+1}. Train {row['train_idx']}: {row['image_path'].name}")
            print(f"   Class: {row['class_name']}")
            print(f"   Appearances: {row['appearance_count']}")
            print(f"   Max influence: {row['max_influence']:.6e}")
            print(f"   Mean influence: {row['mean_influence']:.6e}")
            print(f"   Impact: Appears in {100*row['appearance_count']/self.values.shape[0]:.1f}% of test samples")
        
        # Top harmful/least helpful images
        print("\n" + "-"*80)
        print("LEAST INFLUENTIAL IMAGES (Lowest Influence)")
        print("-"*80)
        
        for idx, row in harmful_df.head(10).iterrows():
            print(f"\n#{idx+1}. Train {row['train_idx']}: {row['image_path'].name}")
            print(f"   Class: {row['class_name']}")
            print(f"   Appearances: {row['appearance_count']}")
            print(f"   Min influence: {row['min_influence']:.6e}")
            print(f"   Mean influence: {row['mean_influence']:.6e}")
        
        print("\n" + "="*80)
    
    def analyze_per_class_influences(self):
        """Analyze influence distribution per class."""
        stats_df = self.analyze_global_influences()
        
        print("\n" + "="*80)
        print("PER-CLASS INFLUENCE ANALYSIS")
        print("="*80 + "\n")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_data = stats_df[stats_df['label'] == class_idx]
            
            if len(class_data) == 0:
                continue
            
            print(f"\n{class_name.upper()}:")
            print(f"  Images in top-K: {len(class_data)}")
            print(f"  Total appearances: {class_data['appearance_count'].sum()}")
            print(f"  Avg appearances per image: {class_data['appearance_count'].mean():.1f}")
            print(f"  Max influence: {class_data['max_influence'].max():.6e}")
            print(f"  Mean influence: {class_data['mean_influence'].mean():.6e}")
            
            # Top 3 most influential images in this class
            top_3 = class_data.nlargest(3, 'max_influence')
            print(f"  Top 3 images:")
            for _, row in top_3.iterrows():
                print(f"    - {row['image_path'].name}: {row['max_influence']:.2e} "
                      f"({row['appearance_count']} appearances)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Inspect top influential training images')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top images to show')
    parser.add_argument('--helpful_only', action='store_true', help='Only show helpful images')
    parser.add_argument('--harmful_only', action='store_true', help='Only show harmful/least influential images')
    parser.add_argument('--show_per_class', action='store_true', help='Show per-class analysis')
    parser.add_argument('--output_helpful', type=str, default='outputs/inspection/top_helpful_images.png',
                       help='Output path for helpful images')
    parser.add_argument('--output_harmful', type=str, default='outputs/inspection/top_harmful_images.png',
                       help='Output path for harmful images')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TOP INFLUENTIAL IMAGES INSPECTOR")
    print("="*80 + "\n")
    
    # Initialize inspector
    inspector = TopInfluenceInspector()
    
    # Get top helpful images
    helpful_df = inspector.get_top_helpful(n=args.top_n)
    
    # Get top harmful images
    harmful_df = inspector.get_top_harmful(n=args.top_n)
    
    # Print report
    inspector.print_influence_report(helpful_df, harmful_df)
    
    # Per-class analysis
    if args.show_per_class:
        inspector.analyze_per_class_influences()
    
    # Visualize
    if not args.harmful_only:
        inspector.visualize_influences(
            helpful_df,
            f'Top {args.top_n} Most Helpful Training Images\n(Highest Positive Influence)',
            args.output_helpful,
            metric='max_influence'
        )
    
    if not args.helpful_only:
        inspector.visualize_influences(
            harmful_df,
            f'Top {args.top_n} Least Influential Training Images\n(Lowest Influence)',
            args.output_harmful,
            metric='min_influence'
        )
    
    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
