"""
TracIn Results Analysis

Visualizes TracIn influence scores with dashboards and extracts top influential images.

Usage:
    python analyze_results.py
    
Output:
    - results/analysis_dashboard.png
    - results/top_influences/{class}/positive|negative/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import shutil

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

class TracInAnalyzer:
    """TracIn results analyzer with visualizations and image extraction."""
    
    def __init__(self, results_dir='results', data_dir='../data/processed'):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = self.results_dir / 'top_influences'
        
        print("Loading TracIn results...")
        self.df = pd.read_csv(self.results_dir / 'influence_summary.csv')
        self.scores = np.load(self.results_dir / 'influence_scores.npy')
        
        print(f"Loaded {self.scores.shape[0]} training × {self.scores.shape[1]} test samples")
        
    def create_dashboard(self):
        """Create visualization dashboard with 3x3 grid layout."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        fig.suptitle('TracIn Influence Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_overview(ax1)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_overview(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_positive_distribution(ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_negative_distribution(ax3)
        
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_average_influence(ax4)
        
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_influence_scatter(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_top_training_samples(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_key_findings(ax7)
        output_path = self.results_dir / 'analysis_dashboard.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved dashboard: {output_path}")
        plt.close()
        
    def _plot_overview(self, ax):
        """Display overview statistics."""
        ax.axis('off')
        
        num_train = self.scores.shape[0]
        num_test = self.scores.shape[1]
        
        stats_text = f"""
        OVERVIEW STATISTICS
        {'='*50}
        
        Dataset Size:
          • Training samples: {num_train:,}
          • Test samples: {num_test:,}
          • Total influence scores: {num_train * num_test:,}
        
        Influence Range:
          • Max positive: {self.df['max_positive_influence'].max():.6f}
          • Max negative: {self.df['max_negative_influence'].min():.6f}
          • Avg positive: {self.df['max_positive_influence'].mean():.6f}
          • Avg negative: {self.df['max_negative_influence'].mean():.6f}
        
        Overall Health:
          • Tests with positive avg: {(self.df['avg_influence'] > 0).sum()} / {num_test}
          • Health score: {(self.df['avg_influence'] > 0).sum() / num_test * 100:.1f}%
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
               
    def _plot_positive_distribution(self, ax):
        """Histogram of positive influences with mean line."""
        mean_val = self.df['max_positive_influence'].mean()
        ax.hist(self.df['max_positive_influence'], bins=30, 
               color='green', alpha=0.7, edgecolor='black')
        ax.axvline(mean_val, 
                  color='red', linestyle='--', linewidth=2, 
                  label=f'Mean = {mean_val:.6f}')
        ax.set_xlabel('Max Positive Influence', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Positive Influence\n(Helpful Training)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_negative_distribution(self, ax):
        """Histogram of negative influences with mean line."""
        mean_val = self.df['max_negative_influence'].mean()
        ax.hist(self.df['max_negative_influence'], bins=30,
               color='red', alpha=0.7, edgecolor='black')
        ax.axvline(mean_val,
                  color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean = {mean_val:.6f}')
        ax.set_xlabel('Max Negative Influence', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Negative Influence\n(Harmful Training)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    def _plot_average_influence(self, ax):
        """Bar chart of average influence per test sample."""
        colors = ['green' if x > 0 else 'red' for x in self.df['avg_influence']]
        bars = ax.bar(self.df['test_idx'], self.df['avg_influence'], 
                     color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Test Sample Index', fontsize=10)
        ax.set_ylabel('Average Influence', fontsize=10)
        ax.set_title('Average Influence per Test Sample (Green=Good, Red=Bad)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        positive_pct = (self.df['avg_influence'] > 0).sum() / len(self.df) * 100
        ax.text(0.98, 0.97, f'{positive_pct:.1f}% positive', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if positive_pct > 80 else 'yellow', alpha=0.5))
        
    def _plot_influence_scatter(self, ax):
        """Scatter plot of positive vs negative influence balance."""
        scatter = ax.scatter(self.df['max_positive_influence'], 
                           self.df['max_negative_influence'],
                           c=self.df['avg_influence'], cmap='RdYlGn',
                           s=80, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Max Positive →', fontsize=9)
        ax.set_ylabel('Max Negative →', fontsize=9)
        ax.set_title('Influence Balance', fontsize=10, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Avg', fraction=0.046, pad=0.04)
        ax.grid(True, alpha=0.3)
        
    def _plot_top_training_samples(self, ax):
        """Horizontal bar chart of top 10 most helpful training samples."""
        train_counts = self.df['most_helpful_train_idx'].value_counts().head(10)
        y_pos = range(len(train_counts))
        ax.barh(y_pos, train_counts.values, color='green', alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Train #{int(idx)}' for idx in train_counts.index], fontsize=8)
        ax.set_xlabel('Times Ranked #1', fontsize=9)
        ax.set_title('Top 10 Most Helpful\nTraining Samples', fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
    def _plot_key_findings(self, ax):
        """Display key findings and health status."""
        ax.axis('off')
        
        top_train = int(self.df['most_helpful_train_idx'].mode()[0])
        top_count = (self.df['most_helpful_train_idx'] == top_train).sum()
        health_pct = (self.df['avg_influence'] > 0).sum() / len(self.df) * 100
        if health_pct > 95:
            status = "EXCELLENT"
            color = 'lightgreen'
        elif health_pct > 80:
            status = "GOOD"
            color = 'lightgreen'
        elif health_pct > 60:
            status = "FAIR"
            color = 'yellow'
        else:
            status = "NEEDS ATTENTION"
            color = 'lightcoral'
        
        findings = f"""
        KEY FINDINGS
        {'='*40}
        
        Health: {status} ({health_pct:.1f}% positive)
        
        {"Quality is good" if health_pct > 80 else "Review data quality"}
        {"No issues detected" if health_pct > 80 else "Check for mislabels"}
        
        """
        
        ax.text(0.05, 0.95, findings, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    def save_top_influential_images(self, top_k=10):
        """Extract and save top k influential training images per class."""
        print("\nSaving top influential images...")
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        
        train_dir = self.data_dir / 'train'
        test_classes = self.df['test_class'].unique()
        
        for test_class in test_classes:
            print(f"\nProcessing class: {test_class}")
            class_data = self.df[self.df['test_class'] == test_class]
            
            pos_dir = self.output_dir / test_class / 'positive'
            neg_dir = self.output_dir / test_class / 'negative'
            pos_dir.mkdir(parents=True, exist_ok=True)
            neg_dir.mkdir(parents=True, exist_ok=True)
            top_positive_indices = class_data.nlargest(top_k, 'max_positive_influence')['most_helpful_train_idx'].values
            top_negative_indices = class_data.nsmallest(top_k, 'max_negative_influence')['most_harmful_train_idx'].values
            print(f"  Saving top {top_k} positive influences...")
            self._save_train_images(top_positive_indices, train_dir, pos_dir, 'positive')
            
            print(f"  Saving top {top_k} negative influences...")
            self._save_train_images(top_negative_indices, train_dir, neg_dir, 'negative')
            
            self._create_class_summary(test_class, class_data, top_positive_indices, top_negative_indices)
        
        print(f"\n✓ Saved top influential images to: {self.output_dir}")
    
    def _save_train_images(self, train_indices, train_dir, output_dir, influence_type):
        """Copy training images to output directory."""
        all_train_images = []
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                for img_path in sorted(class_dir.glob('*')):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        all_train_images.append(img_path)
        
        for rank, idx in enumerate(train_indices, 1):
            idx = int(idx)
            if idx < len(all_train_images):
                src = all_train_images[idx]
                dst = output_dir / f'rank_{rank:02d}_train_{idx:04d}_{src.stem}{src.suffix}'
                shutil.copy2(src, dst)
    
    def _create_class_summary(self, test_class, class_data, top_pos, top_neg):
        """Create 2x3 summary visualization for a specific class."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Influence Analysis: {test_class.upper()}', fontsize=16, fontweight='bold')
        mean_pos = class_data['max_positive_influence'].mean()
        axes[0, 0].hist(class_data['max_positive_influence'], bins=20, 
                       color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(mean_pos, color='red', linestyle='--', linewidth=2,
                          label=f'Mean = {mean_pos:.6f}')
        axes[0, 0].set_xlabel('Max Positive Influence')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Positive Influences')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        mean_neg = class_data['max_negative_influence'].mean()
        axes[0, 1].hist(class_data['max_negative_influence'], bins=20,
                       color='red', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(mean_neg, color='blue', linestyle='--', linewidth=2,
                          label=f'Mean = {mean_neg:.6f}')
        axes[0, 1].set_xlabel('Max Negative Influence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Negative Influences')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        colors = ['green' if x > 0 else 'red' for x in class_data['avg_influence']]
        axes[0, 2].bar(range(len(class_data)), class_data['avg_influence'].values,
                      color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0, 2].set_xlabel('Test Sample Index')
        axes[0, 2].set_ylabel('Average Influence')
        axes[0, 2].set_title('Average Influence per Test Sample')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        positive_pct = (class_data['avg_influence'] > 0).sum() / len(class_data) * 100
        axes[0, 2].text(0.98, 0.97, f'{positive_pct:.1f}% positive',
                       transform=axes[0, 2].transAxes, fontsize=9,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', 
                                facecolor='lightgreen' if positive_pct > 80 else 'yellow', 
                                alpha=0.5))
        unique_pos, counts_pos = np.unique(top_pos, return_counts=True)
        sorted_idx = np.argsort(counts_pos)[::-1][:10]
        axes[1, 0].barh(range(len(sorted_idx)), counts_pos[sorted_idx], 
                       color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_yticks(range(len(sorted_idx)))
        axes[1, 0].set_yticklabels([f'Train #{int(unique_pos[i])}' for i in sorted_idx])
        axes[1, 0].set_xlabel('Frequency in Top Positive')
        axes[1, 0].set_title('Most Frequently Helpful Training Samples')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        unique_neg, counts_neg = np.unique(top_neg, return_counts=True)
        sorted_idx_neg = np.argsort(counts_neg)[::-1][:10]
        axes[1, 1].barh(range(len(sorted_idx_neg)), counts_neg[sorted_idx_neg],
                       color='red', alpha=0.7, edgecolor='black')
        axes[1, 1].set_yticks(range(len(sorted_idx_neg)))
        axes[1, 1].set_yticklabels([f'Train #{int(unique_neg[i])}' for i in sorted_idx_neg])
        axes[1, 1].set_xlabel('Frequency in Top Negative')
        axes[1, 1].set_title('Most Frequently Harmful Training Samples')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 2].axis('off')
        stats_text = f"""
        SUMMARY STATISTICS
        
        Test Samples: {len(class_data)}
        
        Positive Influence:
          Mean: {class_data['max_positive_influence'].mean():.6f}
          Max:  {class_data['max_positive_influence'].max():.6f}
          
        Negative Influence:
          Mean: {class_data['max_negative_influence'].mean():.6f}
          Min:  {class_data['max_negative_influence'].min():.6f}
          
        Average Influence:
          Mean: {class_data['avg_influence'].mean():.6f}
          All Positive: {'Yes' if (class_data['avg_influence'] > 0).all() else 'No'}
        
        Most Influential Training Sample:
          Train #{int(unique_pos[sorted_idx[0]])}
          (appeared {counts_pos[sorted_idx[0]]} times)
        """
        axes[1, 2].text(0.1, 0.9, stats_text, fontsize=11, 
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        summary_path = self.output_dir / test_class / f'{test_class}_summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created summary: {summary_path}")
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*70)
        print("TRACIN ANALYSIS SUMMARY")
        print("="*70)
        
        health_pct = (self.df['avg_influence'] > 0).sum() / len(self.df) * 100
        print(f"\nTraining Data Health: {health_pct:.1f}% of test samples benefit")
        
        if health_pct > 95:
            print("Status: EXCELLENT")
        elif health_pct > 80:
            print("Status: GOOD")
        elif health_pct > 60:
            print("Status: FAIR")
        else:
            print("Status: NEEDS ATTENTION")
        
        top_train = int(self.df['most_helpful_train_idx'].mode()[0])
        top_count = (self.df['most_helpful_train_idx'] == top_train).sum()
        
        print(f"\nStar Training Sample: Train #{top_train}")
        print(f"  → Helped {int(top_count)}/{len(self.df)} test samples ({top_count/len(self.df)*100:.1f}%)")
        
        print(f"\nInfluence Statistics:")
        print(f"  Max Positive:  {self.df['max_positive_influence'].max():.6f}")
        print(f"  Max Negative:  {self.df['max_negative_influence'].min():.6f}")
        print(f"  Avg Positive:  {self.df['max_positive_influence'].mean():.6f}")
        print(f"  Avg Negative:  {self.df['max_negative_influence'].mean():.6f}")
        
        print("\n" + "="*70)
        print("OUTPUT FILES:")
        print("="*70)
        print(f"  1. results/analysis_dashboard.png - Main visualization")
        print(f"  2. results/top_influences/ - Top influential images by class")
        print(f"  3. results/influence_summary.csv - Detailed per-test data")
        print("="*70 + "\n")


def main():
    print("="*70)
    print("TracIn Results Analysis")
    print("="*70)
    
    analyzer = TracInAnalyzer()
    print("\nCreating analysis dashboard...")
    analyzer.create_dashboard()
    
    # Save top influential images
    print("\nExtracting top influential images...")
    analyzer.save_top_influential_images(top_k=10)
    
    # Print summary
    analyzer.print_summary()
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()
