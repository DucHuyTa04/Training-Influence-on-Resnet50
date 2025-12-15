"""
Cross-Version Analysis

Generates comprehensive cross-version analysis of model outputs (v2-v10).
Produces CSVs and dashboards to back up REPORT.md Section 11 findings.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from version_manager import VersionManager

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class CrossVersionAnalyzer:
    """Analyzes patterns across multiple model versions."""
    
    def __init__(self, base_dir, versions=None):
        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / 'outputs'
        self.data_dir = self.base_dir / 'data' / 'processed'
        
        self.vm = VersionManager(self.base_dir)
        
        if versions is None:
            versions = list(range(2, 11))
        self.versions = versions
        
        self._load_class_names()
        self._load_all_version_data()
    
    def _load_class_names(self):
        train_dir = self.data_dir / 'train'
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
    def _load_all_version_data(self):
        self.mispredictions = {}
        self.influence_reports = {}
        self.influence_values = {}
        self.influence_indices = {}
        
        for v in self.versions:
            version_dir = self.outputs_dir / f'v{v}'
            
            mispred_path = version_dir / 'mispredictions' / 'false_predictions.csv'
            if mispred_path.exists():
                self.mispredictions[v] = pd.read_csv(mispred_path)
            
            influence_dir = version_dir / 'influence_analysis'
            values_path = influence_dir / 'top_k_influences_values.npy'
            indices_path = influence_dir / 'top_k_influences_indices.npy'
            
            if values_path.exists() and indices_path.exists():
                self.influence_values[v] = np.load(values_path)
                self.influence_indices[v] = np.load(indices_path)
        
        print(f"[INFO] Loaded data for {len(self.mispredictions)} versions")
    
    def analyze_persistent_mispredictions(self):
        """Find images mispredicted across all versions (Section 11.2)."""
        mispred_counts = defaultdict(lambda: {'versions': [], 'predictions': []})
        
        for v, df in self.mispredictions.items():
            train_df = df[df['split'] == 'train']
            for _, row in train_df.iterrows():
                filename = Path(row['path']).name
                mispred_counts[filename]['versions'].append(v)
                mispred_counts[filename]['predictions'].append(row['pred'])
                mispred_counts[filename]['true'] = row['true']
                mispred_counts[filename]['path'] = row['path']
        
        results = []
        for filename, data in mispred_counts.items():
            if len(data['versions']) >= 7:
                pred_counts = Counter(data['predictions'])
                most_common_pred = pred_counts.most_common(1)[0][0]
                
                results.append({
                    'filename': filename,
                    'true_label': data['true'],
                    'true_class': self.class_names[data['true']],
                    'predicted_label': most_common_pred,
                    'predicted_class': self.class_names[most_common_pred],
                    'versions_affected': len(data['versions']),
                    'version_list': ','.join(map(str, sorted(data['versions']))),
                    'all_predictions': ','.join(map(str, data['predictions'])),
                    'path': data['path']
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values(['versions_affected', 'true_class'], ascending=[False, True])
        return df
    
    def analyze_influential_samples(self):
        """Find consistently influential training samples (Section 11.4)."""
        all_appearances = defaultdict(lambda: {'counts': [], 'versions': []})
        
        for v, indices in self.influence_indices.items():
            idx_counts = Counter(indices.flatten())
            for train_idx, count in idx_counts.items():
                all_appearances[train_idx]['counts'].append(count)
                all_appearances[train_idx]['versions'].append(v)
        
        results = []
        for train_idx, data in all_appearances.items():
            if len(data['versions']) >= 7:
                avg_count = np.mean(data['counts'])
                results.append({
                    'train_idx': train_idx,
                    'avg_appearances': avg_count,
                    'min_appearances': min(data['counts']),
                    'max_appearances': max(data['counts']),
                    'versions_in_top': len(data['versions']),
                    'version_list': ','.join(map(str, sorted(data['versions']))),
                    'appearance_list': ','.join(map(str, data['counts']))
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('avg_appearances', ascending=False)
        
        train_dir = self.data_dir / 'train'
        train_images = []
        train_labels = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = train_dir / class_name
            images = sorted([f for f in class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            train_images.extend(images)
            train_labels.extend([class_idx] * len(images))
        
        def get_class(idx):
            if idx < len(train_labels):
                return self.class_names[train_labels[idx]]
            return 'unknown'
        
        def get_path(idx):
            if idx < len(train_images):
                return str(train_images[idx])
            return ''
        
        df['class'] = df['train_idx'].apply(get_class)
        df['image_path'] = df['train_idx'].apply(get_path)
        
        return df
    
    def analyze_influence_statistics(self):
        """Compute influence statistics per version (Section 11.6)."""
        results = []
        
        for v, values in self.influence_values.items():
            flat = values.flatten()
            results.append({
                'version': v,
                'mean_influence': flat.mean(),
                'median_influence': np.median(flat),
                'std_influence': flat.std(),
                'min_influence': flat.min(),
                'max_influence': flat.max(),
                'negative_count': (flat < 0).sum(),
                'negative_pct': 100 * (flat < 0).sum() / len(flat),
                'total_scores': len(flat)
            })
        
        return pd.DataFrame(results)
    
    def analyze_confusion_patterns(self):
        """Analyze class confusion patterns across versions."""
        confusion_counts = defaultdict(int)
        
        for v, df in self.mispredictions.items():
            for _, row in df.iterrows():
                true_class = self.class_names[row['true']]
                pred_class = self.class_names[row['pred']]
                confusion_counts[(true_class, pred_class)] += 1
        
        results = []
        for (true_class, pred_class), count in confusion_counts.items():
            results.append({
                'true_class': true_class,
                'predicted_class': pred_class,
                'total_count': count,
                'avg_per_version': count / len(self.versions)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('total_count', ascending=False)
        return df
    
    def analyze_class_error_rates(self):
        """Analyze per-class error counts."""
        class_errors = defaultdict(list)
        
        for v, df in self.mispredictions.items():
            version_counts = Counter(df['true'].values)
            for class_idx, class_name in enumerate(self.class_names):
                class_errors[class_name].append(version_counts.get(class_idx, 0))
        
        results = []
        for class_name, counts in class_errors.items():
            results.append({
                'class': class_name,
                'avg_errors': np.mean(counts),
                'min_errors': min(counts),
                'max_errors': max(counts),
                'std_errors': np.std(counts),
                'error_counts': ','.join(map(str, counts))
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('avg_errors', ascending=False)
        return df
    
    def generate_dashboard(self, output_path):
        """Generate comprehensive cross-version dashboard."""
        fig = plt.figure(figsize=(20, 24))
        
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)
        
        # 1. Version Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        registry = self.vm._load_registry()
        versions = sorted([int(k) for k in registry.keys()])
        val_accs = [registry[str(v)].get('val_accuracy', 0) * 100 for v in versions]
        test_accs = [registry[str(v)].get('test_accuracy', 0) * 100 for v in versions]
        
        x = np.arange(len(versions))
        width = 0.35
        ax1.bar(x - width/2, val_accs, width, label='Validation', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, test_accs, width, label='Test', color='darkorange', alpha=0.8)
        ax1.set_xlabel('Version')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Across Versions')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'v{v}' for v in versions])
        ax1.legend()
        ax1.set_ylim(97, 100)
        ax1.axhline(y=99, color='green', linestyle='--', alpha=0.5, label='99% threshold')
        
        # 2. Misprediction Counts
        ax2 = fig.add_subplot(gs[0, 1])
        mispred_counts = []
        for v in self.versions:
            if v in self.mispredictions:
                mispred_counts.append(len(self.mispredictions[v]))
            else:
                mispred_counts.append(0)
        
        ax2.bar([f'v{v}' for v in self.versions], mispred_counts, color='coral', alpha=0.8)
        ax2.set_xlabel('Version')
        ax2.set_ylabel('Total Mispredictions')
        ax2.set_title('Misprediction Count per Version')
        ax2.axhline(y=np.mean(mispred_counts), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(mispred_counts):.1f}')
        ax2.legend()
        
        # 3. Confusion Pattern Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        confusion_df = self.analyze_confusion_patterns()
        
        confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)))
        for _, row in confusion_df.iterrows():
            true_idx = self.class_to_idx[row['true_class']]
            pred_idx = self.class_to_idx[row['predicted_class']]
            confusion_matrix[true_idx, pred_idx] = row['total_count']
        
        sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax3)
        ax3.set_xlabel('Predicted Class')
        ax3.set_ylabel('True Class')
        ax3.set_title('Cross-Version Confusion Matrix (Sum Across All Versions)')
        
        # 4. Class Error Rates
        ax4 = fig.add_subplot(gs[2, 0])
        class_errors = self.analyze_class_error_rates()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_errors)))
        ax4.barh(class_errors['class'], class_errors['avg_errors'], color=colors, alpha=0.8)
        ax4.set_xlabel('Average Errors per Version')
        ax4.set_ylabel('Class')
        ax4.set_title('Average Mispredictions by Class')
        ax4.invert_yaxis()
        
        # 5. Influence Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        inf_stats = self.analyze_influence_statistics()
        
        ax5.bar([f'v{v}' for v in inf_stats['version']], inf_stats['negative_pct'], 
               color='purple', alpha=0.7)
        ax5.set_xlabel('Version')
        ax5.set_ylabel('Negative Influence %')
        ax5.set_title('Percentage of Negative Influences per Version')
        ax5.axhline(y=inf_stats['negative_pct'].mean(), color='red', linestyle='--', 
                   alpha=0.7, label=f'Mean: {inf_stats["negative_pct"].mean():.1f}%')
        ax5.legend()
        
        # 6. Persistent Mispredictions
        ax6 = fig.add_subplot(gs[3, 0])
        persistent = self.analyze_persistent_mispredictions()
        
        persistent_by_class = persistent.groupby('true_class').size().sort_values(ascending=False)
        ax6.barh(persistent_by_class.index, persistent_by_class.values, color='darkred', alpha=0.7)
        ax6.set_xlabel('Number of Persistently Mispredicted Images')
        ax6.set_ylabel('True Class')
        ax6.set_title(f'Persistent Mispredictions by Class (≥7 versions, n={len(persistent)})')
        ax6.invert_yaxis()
        
        # 7. Top Influential Samples
        ax7 = fig.add_subplot(gs[3, 1])
        influential = self.analyze_influential_samples().head(20)
        
        influential_by_class = influential.groupby('class').size().sort_values(ascending=False)
        ax7.barh(influential_by_class.index, influential_by_class.values, color='darkgreen', alpha=0.7)
        ax7.set_xlabel('Count in Top-20 Influential')
        ax7.set_ylabel('Class')
        ax7.set_title('Top-20 Most Influential Samples by Class')
        ax7.invert_yaxis()
        
        plt.suptitle('Cross-Version Analysis Dashboard (v2-v10)', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[SAVED] Dashboard: {output_path}")
    
    def generate_summary_report(self, output_path):
        """Generate text summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CROSS-VERSION ANALYSIS SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append(f"\nVersions analyzed: {', '.join(map(str, self.versions))}")
        lines.append(f"Total versions: {len(self.versions)}")
        
        lines.append("\n" + "-" * 70)
        lines.append("PERSISTENT MISPREDICTIONS")
        lines.append("-" * 70)
        persistent = self.analyze_persistent_mispredictions()
        lines.append(f"Total persistent mispredictions (≥7 versions): {len(persistent)}")
        lines.append(f"Mispredicted in ALL versions: {len(persistent[persistent['versions_affected'] == len(self.versions)])}")
        
        lines.append("\n" + "-" * 70)
        lines.append("TOP CONFUSION PAIRS")
        lines.append("-" * 70)
        confusion = self.analyze_confusion_patterns().head(10)
        for _, row in confusion.iterrows():
            lines.append(f"  {row['true_class']:>12} → {row['predicted_class']:<12}: {row['total_count']:4} total ({row['avg_per_version']:.1f}/version)")
        
        lines.append("\n" + "-" * 70)
        lines.append("CLASS ERROR RATES")
        lines.append("-" * 70)
        class_errors = self.analyze_class_error_rates()
        for _, row in class_errors.iterrows():
            lines.append(f"  {row['class']:>12}: {row['avg_errors']:5.1f} avg errors (range: {row['min_errors']:.0f}-{row['max_errors']:.0f})")
        
        lines.append("\n" + "-" * 70)
        lines.append("INFLUENCE STATISTICS")
        lines.append("-" * 70)
        inf_stats = self.analyze_influence_statistics()
        for _, row in inf_stats.iterrows():
            lines.append(f"  v{row['version']}: mean={row['mean_influence']:.2e}, range=[{row['min_influence']:.4f}, {row['max_influence']:.4f}], neg={row['negative_pct']:.1f}%")
        
        lines.append("\n" + "=" * 70)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"[SAVED] Summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cross-version analysis of model outputs')
    parser.add_argument('--versions', type=str, default='2-10', help='Version range (e.g., "2-10" or "2,3,5,7")')
    parser.add_argument('--output_dir', type=str, default='outputs/cross_version_analysis', help='Output directory')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    
    if '-' in args.versions:
        start, end = map(int, args.versions.split('-'))
        versions = list(range(start, end + 1))
    else:
        versions = [int(v.strip()) for v in args.versions.split(',')]
    
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Analyzing versions: {versions}")
    print(f"[INFO] Output directory: {output_dir}")
    
    analyzer = CrossVersionAnalyzer(script_dir, versions=versions)
    
    # Generate CSVs
    print("\n[GENERATING] CSV reports...")
    
    persistent_df = analyzer.analyze_persistent_mispredictions()
    persistent_df.to_csv(output_dir / 'persistent_mispredictions.csv', index=False)
    print(f"  → persistent_mispredictions.csv ({len(persistent_df)} rows)")
    
    influential_df = analyzer.analyze_influential_samples()
    influential_df.to_csv(output_dir / 'influential_samples.csv', index=False)
    print(f"  → influential_samples.csv ({len(influential_df)} rows)")
    
    inf_stats_df = analyzer.analyze_influence_statistics()
    inf_stats_df.to_csv(output_dir / 'influence_statistics.csv', index=False)
    print(f"  → influence_statistics.csv ({len(inf_stats_df)} rows)")
    
    confusion_df = analyzer.analyze_confusion_patterns()
    confusion_df.to_csv(output_dir / 'confusion_patterns.csv', index=False)
    print(f"  → confusion_patterns.csv ({len(confusion_df)} rows)")
    
    class_errors_df = analyzer.analyze_class_error_rates()
    class_errors_df.to_csv(output_dir / 'class_error_rates.csv', index=False)
    print(f"  → class_error_rates.csv ({len(class_errors_df)} rows)")
    
    # Generate dashboard
    print("\n[GENERATING] Dashboard...")
    analyzer.generate_dashboard(output_dir / 'cross_version_dashboard.png')
    
    # Generate summary report
    print("\n[GENERATING] Summary report...")
    analyzer.generate_summary_report(output_dir / 'summary_report.txt')
    
    print("\n[DONE] Cross-version analysis complete!")
    print(f"[OUTPUT] All files saved to: {output_dir}")


if __name__ == '__main__':
    main()
