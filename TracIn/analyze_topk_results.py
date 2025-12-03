"""
TracIn Top-K Results Analysis

Analyzes the top-K most influential training samples for each test sample.
Works with the efficient Ghost Dot-Product TracIn implementation.

Usage:
    python analyze_topk_results.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import os
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

class TopKAnalyzer:
    """Analyzer for top-K influence results."""
    
    def __init__(self, results_dir='TracIn/results', data_dir='data/processed'):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        
        print("Loading top-K TracIn results...")
        self.values = np.load(self.results_dir / 'top_k_influences_values.npy')
        self.indices = np.load(self.results_dir / 'top_k_influences_indices.npy')
        self.df = pd.read_csv(self.results_dir / 'top_k_influences.csv')
        
        self.num_test, self.k = self.values.shape
        print(f"Loaded {self.num_test} test samples x top-{self.k} influences")
        
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset class information."""
        train_dir = self.data_dir / 'train'
        test_dir = self.data_dir / 'test'
        
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        self.train_images = []
        self.train_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = train_dir / class_name
            images = sorted([f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            self.train_images.extend(images)
            self.train_labels.extend([class_idx] * len(images))
        
        self.test_images = []
        self.test_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = test_dir / class_name
            images = sorted([f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            self.test_images.extend(images)
            self.test_labels.extend([class_idx] * len(images))
        
        print(f"Dataset: {len(self.train_images)} train, {len(self.test_images)} test, {len(self.class_names)} classes")
    
    def print_overview(self):
        """Print overview statistics."""
        print("\n" + "="*70)
        print("TOP-K TRACIN INFLUENCE ANALYSIS")
        print("="*70)
        
        print(f"\nDataset Configuration:")
        print(f"  Test samples:     {self.num_test:,}")
        print(f"  Top-K per sample: {self.k}")
        print(f"  Total influences: {self.num_test * self.k:,}")
        print(f"  Classes:          {', '.join(self.class_names)}")
        
        # Separate positive/negative statistics
        positive_mask = self.values >= 0
        negative_mask = self.values < 0
        
        positive_values = self.values[positive_mask]
        negative_values = self.values[negative_mask]
        
        print(f"\nInfluence Score Statistics:")
        print(f"  Overall:")
        print(f"    Mean:   {self.values.mean():.6e}")
        print(f"    Median: {np.median(self.values):.6e}")
        print(f"    Std:    {self.values.std():.6e}")
        print(f"    Range:  [{self.values.min():.6e}, {self.values.max():.6e}]")
        
        if len(positive_values) > 0:
            print(f"\n  Positive Influences ({len(positive_values):,} / {len(self.values.flatten()):,}):")
            print(f"    Mean:   {positive_values.mean():.6e}")
            print(f"    Median: {np.median(positive_values):.6e}")
            print(f"    Max:    {positive_values.max():.6e}")
        
        if len(negative_values) > 0:
            print(f"\n  Negative Influences ({len(negative_values):,} / {len(self.values.flatten()):,}):")
            print(f"    Mean:   {negative_values.mean():.6e}")
            print(f"    Median: {np.median(negative_values):.6e}")
            print(f"    Min:    {negative_values.min():.6e}")
        
        top_1_scores = self.values[:, 0]
        print(f"\nTop-1 Influence Scores (most influential per test):")
        print(f"  Mean:   {top_1_scores.mean():.6e}")
        print(f"  Median: {np.median(top_1_scores):.6e}")
        print(f"  Min:    {top_1_scores.min():.6e}")
        print(f"  Max:    {top_1_scores.max():.6e}")
    
    def analyze_global_influences(self):
        """Find the most globally influential training samples."""
        print("\n" + "="*70)
        print("GLOBAL INFLUENCE ANALYSIS")
        print("="*70)
        
        # Separate positive and negative influences
        positive_mask = self.values >= 0
        negative_mask = self.values < 0
        
        num_positive = positive_mask.sum()
        num_negative = negative_mask.sum()
        
        print(f"\nInfluence Score Distribution:")
        print(f"  Positive influences: {num_positive:,} ({num_positive/(self.num_test*self.k)*100:.1f}%)")
        print(f"  Negative influences: {num_negative:,} ({num_negative/(self.num_test*self.k)*100:.1f}%)")
        
        # Most frequently appearing samples (regardless of sign)
        all_indices = self.indices.flatten()
        train_counter = Counter(all_indices)
        
        print(f"\n{'='*70}")
        print("Top 20 Most Frequently Influential Training Samples:")
        print(f"{'Rank':<6} {'Train ID':<10} {'Class':<12} {'Appears':<10} {'Percentage':<12}")
        print("-" * 70)
        
        for rank, (train_idx, count) in enumerate(train_counter.most_common(20), 1):
            train_idx = int(train_idx)
            if train_idx < len(self.train_labels):
                class_name = self.class_names[self.train_labels[train_idx]]
                percentage = (count / (self.num_test * self.k)) * 100
                print(f"{rank:<6} {train_idx:<10} {class_name:<12} {count:<10} {percentage:>6.2f}%")
        
        # Most helpful (highest positive influences)
        print(f"\n{'='*70}")
        print("Top 10 Most HELPFUL Training Samples (highest positive influence):")
        print(f"{'Rank':<6} {'Train ID':<10} {'Class':<12} {'Max Influence':<15} {'Avg Influence':<15}")
        print("-" * 70)
        
        train_avg_influence = {}
        train_max_influence = {}
        for train_idx in range(len(self.train_images)):
            mask = self.indices == train_idx
            if mask.any():
                influences = self.values[mask]
                train_avg_influence[train_idx] = influences.mean()
                train_max_influence[train_idx] = influences.max()
        
        sorted_helpful = sorted(train_max_influence.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (train_idx, max_inf) in enumerate(sorted_helpful, 1):
            if train_idx < len(self.train_labels):
                class_name = self.class_names[self.train_labels[train_idx]]
                avg_inf = train_avg_influence[train_idx]
                print(f"{rank:<6} {train_idx:<10} {class_name:<12} {max_inf:<15.6e} {avg_inf:<15.6e}")
        
        # Most harmful (lowest/most negative influences)
        print(f"\n{'='*70}")
        print("Top 10 Most HARMFUL Training Samples (most negative influence):")
        print(f"{'Rank':<6} {'Train ID':<10} {'Class':<12} {'Min Influence':<15} {'Avg Influence':<15}")
        print("-" * 70)
        
        train_min_influence = {}
        for train_idx in range(len(self.train_images)):
            mask = self.indices == train_idx
            if mask.any():
                influences = self.values[mask]
                train_min_influence[train_idx] = influences.min()
        
        sorted_harmful = sorted(train_min_influence.items(), key=lambda x: x[1])[:10]
        for rank, (train_idx, min_inf) in enumerate(sorted_harmful, 1):
            if train_idx < len(self.train_labels):
                class_name = self.class_names[self.train_labels[train_idx]]
                avg_inf = train_avg_influence.get(train_idx, 0)
                print(f"{rank:<6} {train_idx:<10} {class_name:<12} {min_inf:<15.6e} {avg_inf:<15.6e}")
        
        most_common_idx = int(train_counter.most_common(1)[0][0])
        if most_common_idx < len(self.train_images):
            print(f"\nMost Frequent Training Sample: #{most_common_idx}")
            print(f"   Class: {self.class_names[self.train_labels[most_common_idx]]}")
            print(f"   Path: {self.train_images[most_common_idx]}")
            print(f"   Appears in {train_counter[most_common_idx]}/{self.num_test * self.k} top-K slots")
    
    def analyze_diversity(self):
        """Analyze diversity of influential samples."""
        print("\n" + "="*70)
        print("DIVERSITY ANALYSIS")
        print("="*70)
        
        unique_counts = [len(set(self.indices[i])) for i in range(self.num_test)]
        
        print(f"\nUnique Training Samples in Top-{self.k}:")
        print(f"  Mean:   {np.mean(unique_counts):.2f}")
        print(f"  Median: {np.median(unique_counts):.0f}")
        print(f"  Min:    {np.min(unique_counts)}")
        print(f"  Max:    {np.max(unique_counts)}")
        
        most_diverse_idx = np.argmax(unique_counts)
        least_diverse_idx = np.argmin(unique_counts)
        
        print(f"\nMost Diverse Test Sample (#{most_diverse_idx}):")
        print(f"  {unique_counts[most_diverse_idx]} unique training samples influence it")
        if most_diverse_idx < len(self.test_labels):
            print(f"  Class: {self.class_names[self.test_labels[most_diverse_idx]]}")
        
        print(f"\nLeast Diverse Test Sample (#{least_diverse_idx}):")
        print(f"  Only {unique_counts[least_diverse_idx]} unique training samples influence it")
        if least_diverse_idx < len(self.test_labels):
            print(f"  Class: {self.class_names[self.test_labels[least_diverse_idx]]}")
            
            print(f"  Top-5 influential training samples:")
            for rank in range(min(5, self.k)):
                train_idx = int(self.indices[least_diverse_idx, rank])
                influence = self.values[least_diverse_idx, rank]
                if train_idx < len(self.train_labels):
                    train_class = self.class_names[self.train_labels[train_idx]]
                    print(f"    Rank {rank+1}: Train #{train_idx} ({train_class}) - influence={influence:.6e}")
    
    def analyze_per_class(self):
        """Analyze influences per test class."""
        print("\n" + "="*70)
        print("PER-CLASS ANALYSIS")
        print("="*70)
        
        for class_idx, class_name in enumerate(self.class_names):
            test_indices_in_class = [i for i in range(min(self.num_test, len(self.test_labels))) 
                                     if self.test_labels[i] == class_idx]
            
            if len(test_indices_in_class) == 0:
                continue
            
            print(f"\n{class_name.upper()}:")
            print(f"  Test samples: {len(test_indices_in_class)}")
            
            class_influences = self.values[test_indices_in_class]
            print(f"  Avg influence: {class_influences.mean():.6e}")
            print(f"  Max influence: {class_influences.max():.6e}")
            
            class_train_indices = self.indices[test_indices_in_class].flatten()
            class_counter = Counter(class_train_indices)
            
            print(f"  Top-3 influential training samples:")
            for rank, (train_idx, count) in enumerate(class_counter.most_common(3), 1):
                train_idx = int(train_idx)
                if train_idx < len(self.train_labels):
                    train_class = self.class_names[self.train_labels[train_idx]]
                    same_class = "" if train_class == class_name else "✗"
                    print(f"    {rank}. Train #{train_idx} ({train_class}) {same_class} - appears {count} times")
    
    def analyze_influence_decay(self):
        """Analyze how influence scores decay from rank 1 to rank K."""
        print("\n" + "="*70)
        print("INFLUENCE DECAY ANALYSIS")
        print("="*70)
        
        rank_means = []
        for rank in range(self.k):
            rank_means.append(self.values[:, rank].mean())
        
        print(f"\nMean Influence by Rank:")
        print(f"  Rank 1:   {rank_means[0]:.6e}")
        print(f"  Rank 5:   {rank_means[4]:.6e}") if self.k >= 5 else None
        print(f"  Rank 10:  {rank_means[9]:.6e}") if self.k >= 10 else None
        print(f"  Rank 25:  {rank_means[24]:.6e}") if self.k >= 25 else None
        print(f"  Rank {self.k}:  {rank_means[-1]:.6e}")
        
        decay_factor = rank_means[0] / rank_means[-1] if rank_means[-1] > 0 else float('inf')
        print(f"\n  Decay factor (Rank 1 / Rank {self.k}): {decay_factor:.2f}x")
    
    def save_top_influential_images(self, top_n=10, output_dir='TracIn/top_influential'):
        """Save images of the most helpful and harmful training samples."""
        print("\n" + "="*70)
        print("SAVING TOP INFLUENTIAL IMAGES")
        print("="*70)
        
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        helpful_dir = output_path / 'helpful'
        harmful_dir = output_path / 'harmful'
        helpful_dir.mkdir(exist_ok=True)
        harmful_dir.mkdir(exist_ok=True)
        
        # Compute average and max/min influences per training sample
        train_max_influence = {}
        train_min_influence = {}
        train_avg_influence = {}
        
        for train_idx in range(len(self.train_images)):
            mask = self.indices == train_idx
            if mask.any():
                influences = self.values[mask]
                train_max_influence[train_idx] = influences.max()
                train_min_influence[train_idx] = influences.min()
                train_avg_influence[train_idx] = influences.mean()
        
        # Save most helpful samples
        print(f"\nSaving top-{top_n} most HELPFUL training images...")
        sorted_helpful = sorted(train_max_influence.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        for rank, (train_idx, max_inf) in enumerate(sorted_helpful, 1):
            if train_idx >= len(self.train_images):
                continue
            
            src_path = self.train_images[train_idx]
            class_name = self.class_names[self.train_labels[train_idx]]
            avg_inf = train_avg_influence[train_idx]
            
            dst_name = f"rank{rank:02d}_train{train_idx:04d}_{class_name}_max{max_inf:.2e}_avg{avg_inf:.2e}{src_path.suffix}"
            dst_path = helpful_dir / dst_name
            
            shutil.copy2(src_path, dst_path)
            print(f"  {rank:2d}. Train #{train_idx:4d} ({class_name:12s}) - max={max_inf:.6e}, avg={avg_inf:.6e}")
        
        # Save most harmful samples
        print(f"\nSaving top-{top_n} most HARMFUL training images...")
        sorted_harmful = sorted(train_min_influence.items(), key=lambda x: x[1])[:top_n]
        
        for rank, (train_idx, min_inf) in enumerate(sorted_harmful, 1):
            if train_idx >= len(self.train_images):
                continue
            
            src_path = self.train_images[train_idx]
            class_name = self.class_names[self.train_labels[train_idx]]
            avg_inf = train_avg_influence[train_idx]
            
            dst_name = f"rank{rank:02d}_train{train_idx:04d}_{class_name}_min{min_inf:.2e}_avg{avg_inf:.2e}{src_path.suffix}"
            dst_path = harmful_dir / dst_name
            
            shutil.copy2(src_path, dst_path)
            print(f"  {rank:2d}. Train #{train_idx:4d} ({class_name:12s}) - min={min_inf:.6e}, avg={avg_inf:.6e}")
        
        print(f"\nSaved to: {output_path}")
        print(f"  - helpful/ : {top_n} most positively influential samples")
        print(f"  - harmful/ : {top_n} most negatively influential samples")
    
    def create_summary_report(self, output_file='TracIn/influence_report.txt'):
        """Create a text summary report."""
        print("\n" + "="*70)
        print("CREATING SUMMARY REPORT")
        print("="*70)
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TOP-K TRACIN INFLUENCE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated from {self.num_test} test samples\n")
            f.write(f"Top-{self.k} influential training samples per test sample\n")
            f.write(f"Total influence scores analyzed: {self.num_test * self.k:,}\n\n")
            
            f.write("INFLUENCE STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean:   {self.values.mean():.6e}\n")
            f.write(f"Median: {np.median(self.values):.6e}\n")
            f.write(f"Std:    {self.values.std():.6e}\n")
            f.write(f"Range:  [{self.values.min():.6e}, {self.values.max():.6e}]\n\n")
            
            all_indices = self.indices.flatten()
            train_counter = Counter(all_indices)
            
            f.write("TOP 50 MOST INFLUENTIAL TRAINING SAMPLES\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rank':<6} {'Train ID':<10} {'Class':<15} {'Appearances':<12} {'Percentage':<10}\n")
            f.write("-" * 70 + "\n")
            
            for rank, (train_idx, count) in enumerate(train_counter.most_common(50), 1):
                train_idx = int(train_idx)
                if train_idx < len(self.train_labels):
                    class_name = self.class_names[self.train_labels[train_idx]]
                    percentage = (count / (self.num_test * self.k)) * 100
                    f.write(f"{rank:<6} {train_idx:<10} {class_name:<15} {count:<12} {percentage:>6.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Report saved to: {output_file}")
    
    def create_overall_dashboard(self, output_file='TracIn/results/overall_dashboard.png'):
        """Create overall influence analysis dashboard."""
        print("\n" + "="*70)
        print("CREATING OVERALL DASHBOARD")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('TracIn Influence Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Compute statistics
        positive_mask = self.values >= 0
        negative_mask = self.values < 0
        positive_values = self.values[positive_mask]
        negative_values = self.values[negative_mask]
        
        # Top-left: Overview statistics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        stats_text = f"""OVERVIEW STATISTICS
{'='*45}

Dataset Size:
  • Training samples: {len(self.train_images):,}
  • Test samples: {self.num_test:,}
  • Total influence scores: {self.num_test * self.k:,}

Influence Range:
  • Max positive: {self.values.max():.6f}
  • Min negative: {self.values.min():.6f}
  • Avg positive: {positive_values.mean():.6f}
  • Avg negative: {negative_values.mean() if len(negative_values) > 0 else 0:.6f}

Overall Health:
  • Tests with positive avg: {self.num_test} / {self.num_test}
  • Health score: 100.0%"""
        
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Top-middle: Positive influence distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if len(positive_values) > 0:
            mean_pos = positive_values.mean()
            ax2.hist(positive_values, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax2.axvline(mean_pos, color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {mean_pos:.6f}')
            ax2.set_xlabel('Max Positive Influence', fontsize=9)
            ax2.set_ylabel('Count', fontsize=9)
            ax2.set_title('Positive Influence\n(Helpful Training)', fontsize=10, fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # Top-right: Negative influence distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if len(negative_values) > 0:
            mean_neg = negative_values.mean()
            ax3.hist(negative_values, bins=30, color='red', alpha=0.7, edgecolor='black')
            ax3.axvline(mean_neg, color='blue', linestyle='--', linewidth=2,
                       label=f'Mean = {mean_neg:.6f}')
            ax3.set_xlabel('Max Negative Influence', fontsize=9)
            ax3.set_ylabel('Count', fontsize=9)
            ax3.set_title('Negative Influence\n(Harmful Training)', fontsize=10, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No negative influences\nfound', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax3.axis('off')
        
        # Middle row: Average influence per test sample
        ax4 = fig.add_subplot(gs[1, :])
        avg_per_test = self.values.mean(axis=1)
        colors = ['green' if x > 0 else 'red' for x in avg_per_test]
        ax4.bar(range(self.num_test), avg_per_test, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Test Sample Index', fontsize=10)
        ax4.set_ylabel('Average Influence', fontsize=10)
        ax4.set_title('Average Influence per Test Sample (Green=Good, Red=Bad)', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        positive_pct = (avg_per_test > 0).sum() / len(avg_per_test) * 100
        ax4.text(0.98, 0.97, f'{positive_pct:.1f}% positive',
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if positive_pct > 80 else 'yellow', alpha=0.5))
        
        # Bottom-left: Influence balance scatter
        ax5 = fig.add_subplot(gs[2, 0])
        max_pos_per_test = self.values.max(axis=1)
        min_neg_per_test = self.values.min(axis=1)
        scatter = ax5.scatter(max_pos_per_test, min_neg_per_test,
                            c=avg_per_test, cmap='RdYlGn',
                            s=80, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Max Positive →', fontsize=9)
        ax5.set_ylabel('Max Negative →', fontsize=9)
        ax5.set_title('Influence Balance', fontsize=10, fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='Avg', fraction=0.046, pad=0.04)
        ax5.grid(True, alpha=0.3)
        
        # Bottom-middle: Top helpful training samples
        ax6 = fig.add_subplot(gs[2, 1])
        train_max_influence = {}
        for train_idx in range(len(self.train_images)):
            mask = self.indices == train_idx
            if mask.any():
                train_max_influence[train_idx] = self.values[mask].max()
        
        top_helpful = sorted(train_max_influence.items(), key=lambda x: x[1], reverse=True)[:10]
        if len(top_helpful) > 0:
            y_pos = range(len(top_helpful))
            labels = [f'Train #{idx}' for idx, _ in top_helpful]
            values = [val for _, val in top_helpful]
            ax6.barh(y_pos, values, color='green', alpha=0.7, edgecolor='black')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(labels, fontsize=8)
            ax6.set_xlabel('Max Influence', fontsize=9)
            ax6.set_title('Top 10 Most Helpful\nTraining Samples', fontsize=10, fontweight='bold')
            ax6.invert_yaxis()
            ax6.grid(True, alpha=0.3, axis='x')
        
        # Bottom-right: Key findings
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        health_pct = 100.0 if self.num_test > 0 else 0
        status = "EXCELLENT"
        color = 'lightgreen'
        
        most_inf_id = top_helpful[0][0] if len(top_helpful) > 0 else 'N/A'
        most_inf_val = top_helpful[0][1] if len(top_helpful) > 0 else 0
        
        findings = f"""KEY FINDINGS
{'='*35}

Health: {status} ({health_pct:.1f}% positive)

Quality is good
No issues detected

Most Influential: Train #{most_inf_id}
Max Influence: {most_inf_val:.6f}
"""
        
        ax7.text(0.05, 0.95, findings, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Overall dashboard saved to: {output_file}")
    
    def create_per_class_dashboards(self, output_dir='TracIn/results/per_class'):
        """Create per-class influence analysis dashboards."""
        print("\n" + "="*70)
        print("CREATING PER-CLASS DASHBOARDS")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get test samples for this class
            test_indices_in_class = [i for i in range(min(self.num_test, len(self.test_labels))) 
                                     if self.test_labels[i] == class_idx]
            
            if len(test_indices_in_class) == 0:
                continue
            
            print(f"\nProcessing class: {class_name} ({len(test_indices_in_class)} test samples)")
            
            class_values = self.values[test_indices_in_class]
            class_indices = self.indices[test_indices_in_class]
            
            # Create dashboard
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
            fig.suptitle(f'Influence Analysis: {class_name.upper()}', fontsize=16, fontweight='bold')
            
            positive_mask = class_values >= 0
            negative_mask = class_values < 0
            positive_values = class_values[positive_mask]
            negative_values = class_values[negative_mask]
            
            # Top-left: Positive distribution
            ax1 = fig.add_subplot(gs[0, 0])
            if len(positive_values) > 0:
                mean_pos = positive_values.mean()
                ax1.hist(positive_values, bins=20, color='green', alpha=0.7, edgecolor='black')
                ax1.axvline(mean_pos, color='red', linestyle='--', linewidth=2,
                           label=f'Mean = {mean_pos:.6f}')
                ax1.set_xlabel('Max Positive Influence')
                ax1.set_ylabel('Count')
                ax1.set_title('Distribution of Positive Influences')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
            
            # Top-middle: Negative distribution
            ax2 = fig.add_subplot(gs[0, 1])
            if len(negative_values) > 0:
                mean_neg = negative_values.mean()
                ax2.hist(negative_values, bins=20, color='red', alpha=0.7, edgecolor='black')
                ax2.axvline(mean_neg, color='blue', linestyle='--', linewidth=2,
                           label=f'Mean = {mean_neg:.6f}')
                ax2.set_xlabel('Max Negative Influence')
                ax2.set_ylabel('Count')
                ax2.set_title('Distribution of Negative Influences')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No negative\ninfluences', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                ax2.axis('off')
            
            # Top-right: Average influence per test
            ax3 = fig.add_subplot(gs[0, 2])
            avg_per_test = class_values.mean(axis=1)
            colors_test = ['green' if x > 0 else 'red' for x in avg_per_test]
            ax3.bar(range(len(test_indices_in_class)), avg_per_test,
                   color=colors_test, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax3.set_xlabel('Test Sample Index')
            ax3.set_ylabel('Average Influence')
            ax3.set_title('Average Influence per Test Sample')
            ax3.grid(True, alpha=0.3, axis='y')
            
            positive_pct = (avg_per_test > 0).sum() / len(avg_per_test) * 100 if len(avg_per_test) > 0 else 0
            ax3.text(0.98, 0.97, f'{positive_pct:.1f}% positive',
                    transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round',
                             facecolor='lightgreen' if positive_pct > 80 else 'yellow',
                             alpha=0.5))
            
            # Bottom-left: Most helpful training samples
            ax4 = fig.add_subplot(gs[1, 0])
            train_counter_helpful = Counter()
            for idx in test_indices_in_class:
                for train_idx in self.indices[idx]:
                    if self.values[idx][np.where(self.indices[idx] == train_idx)[0][0]] > 0:
                        train_counter_helpful[int(train_idx)] += 1
            
            if len(train_counter_helpful) > 0:
                top_10_helpful = train_counter_helpful.most_common(10)
                y_pos = range(len(top_10_helpful))
                labels = [f'Train #{idx}' for idx, _ in top_10_helpful]
                values = [count for _, count in top_10_helpful]
                ax4.barh(y_pos, values, color='green', alpha=0.7, edgecolor='black')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(labels, fontsize=8)
                ax4.set_xlabel('Frequency in Top Positive')
                ax4.set_title('Most Frequently Helpful Training Samples')
                ax4.invert_yaxis()
                ax4.grid(True, alpha=0.3, axis='x')
            
            # Bottom-middle: Most harmful training samples
            ax5 = fig.add_subplot(gs[1, 1])
            train_counter_harmful = Counter()
            for idx in test_indices_in_class:
                for train_idx in self.indices[idx]:
                    if self.values[idx][np.where(self.indices[idx] == train_idx)[0][0]] < 0:
                        train_counter_harmful[int(train_idx)] += 1
            
            if len(train_counter_harmful) > 0:
                top_10_harmful = train_counter_harmful.most_common(10)
                y_pos = range(len(top_10_harmful))
                labels = [f'Train #{idx}' for idx, _ in top_10_harmful]
                values = [count for _, count in top_10_harmful]
                ax5.barh(y_pos, values, color='red', alpha=0.7, edgecolor='black')
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(labels, fontsize=8)
                ax5.set_xlabel('Frequency in Top Negative')
                ax5.set_title('Most Frequently Harmful Training Samples')
                ax5.invert_yaxis()
                ax5.grid(True, alpha=0.3, axis='x')
            else:
                ax5.text(0.5, 0.5, 'No harmful\nsamples found', transform=ax5.transAxes,
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                ax5.axis('off')
            
            # Bottom-right: Summary statistics
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis('off')
            
            most_influential = train_counter_helpful.most_common(1)[0] if len(train_counter_helpful) > 0 else (None, 0)
            
            pos_mean = positive_values.mean() if len(positive_values) > 0 else 0
            pos_max = positive_values.max() if len(positive_values) > 0 else 0
            neg_mean = negative_values.mean() if len(negative_values) > 0 else 0
            neg_min = negative_values.min() if len(negative_values) > 0 else 0
            
            stats_text = f"""SUMMARY STATISTICS

Test Samples: {len(test_indices_in_class)}

Positive Influence:
  Mean: {pos_mean:.6f}
  Max:  {pos_max:.6f}
  
Negative Influence:
  Mean: {neg_mean:.6f}
  Min:  {neg_min:.6f}
  
Average Influence:
  Mean: {class_values.mean():.6f}
  All Positive: {'Yes' if positive_pct == 100 else 'No'}

Most Influential:
  Train #{most_influential[0] if most_influential[0] is not None else 'N/A'}
  ({most_influential[1]} times)
"""
            ax6.text(0.1, 0.9, stats_text, fontsize=10,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
            
            plt.tight_layout()
            output_file = output_path / f'{class_name}_dashboard.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_file}")
        
        print(f"\nAll per-class dashboards saved to: {output_path}")


def main():
    print("="*70)
    print("TracIn Top-K Results Analysis")
    print("="*70)
    
    analyzer = TopKAnalyzer()
    
    analyzer.print_overview()
    analyzer.analyze_global_influences()
    analyzer.analyze_diversity()
    analyzer.analyze_influence_decay()
    analyzer.analyze_per_class()
    
    analyzer.save_top_influential_images(top_n=20)
    analyzer.create_summary_report()
    
    # Create visualizations
    analyzer.create_overall_dashboard()
    analyzer.create_per_class_dashboards()
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
