"""
Evaluation Script - Compare All Retrieval Methods

Compares baseline methods (BM25, Translate-Test) with Dense Retrieval
and generates comprehensive comparison plots.

"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app.evidence import superiority_conclusion  # noqa: E402


class RetrievalEvaluator:
    """
    Evaluates and compares different retrieval methods.
    """
    
    def __init__(self, results_dir='results'):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory containing results from all methods
        """
        self.results_dir = Path(results_dir)
        self.baseline_dir = self.results_dir / 'baseline'
        self.dense_dir = self.results_dir / 'dense'
        self.output_dir = self.results_dir / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_results(self):
        """Load results from all methods."""
        print("Loading evaluation results...")
        
        results = {}
        
        # Load baseline results
        for method in ['bm25', 'translate']:
            result_file = self.baseline_dir / f'{method}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[method] = json.load(f)
                print(f"✓ Loaded {method} results")
        
        # Load dense retrieval results
        for result_file in self.dense_dir.glob('*_results.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)
                model_name = data['model'].split('/')[-1]  # Get short name
                results[f'dense_{model_name}'] = data
                print(f"✓ Loaded {model_name} results")
        
        if not results:
            raise FileNotFoundError("No result files found. Run baseline.py and dense_retrieval.py first.")
        
        return results
    
    def create_comparison_table(self, results):
        """Create comparison table of all methods."""
        print("\nCreating comparison table...")
        
        comparison = []
        
        for method_name, data in results.items():
            row = {
                'Method': method_name.replace('_', ' ').title(),
                'MRR': data['mrr'],
                'Recall@10': data.get('recall@10', data.get('recall@5', 0)),
                'nDCG@10': data.get('ndcg@10', 0)
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('MRR', ascending=False)
        
        # Save to CSV
        output_file = self.output_dir / 'method_comparison.csv'
        df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"✓ Saved comparison table to {output_file}")
        
        # Print table
        print("\n" + "="*70)
        print("METHOD COMPARISON")
        print("="*70)
        print(df.to_string(index=False, float_format='%.4f'))
        print("="*70 + "\n")
        
        return df
    
    def plot_overall_comparison(self, comparison_df):
        """Create bar plot comparing all methods."""
        print("Creating overall comparison plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['MRR', 'Recall@10', 'nDCG@10']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df.plot(
                    x='Method',
                    y=metric,
                    kind='bar',
                    ax=ax,
                    color=color,
                    legend=False
                )
                ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                ax.set_xlabel('Method', fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_ylim(0, 1.0)
                ax.grid(axis='y', alpha=0.3)
                
                # Rotate x labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'overall_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {output_file}")
        
        plt.close()
    
    def plot_per_language_performance(self, results):
        """Create plot showing per-language performance."""
        print("Creating per-language performance plot...")
        
        # Collect per-language MRR data
        data_for_plot = []
        
        for method_name, data in results.items():
            if 'per_language_mrr' in data:
                for lang, mrr in data['per_language_mrr'].items():
                    data_for_plot.append({
                        'Method': method_name.replace('_', ' ').title(),
                        'Language': lang.upper(),
                        'MRR': mrr
                    })
        
        df = pd.DataFrame(data_for_plot)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        languages = sorted(df['Language'].unique())
        methods = sorted(df['Method'].unique())
        
        x = np.arange(len(languages))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            values = [method_data[method_data['Language'] == lang]['MRR'].values[0] 
                     if len(method_data[method_data['Language'] == lang]) > 0 else 0
                     for lang in languages]
            
            ax.bar(x + i * width, values, width, label=method)
        
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('MRR', fontsize=12)
        ax.set_title('Per-Language Performance (MRR)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(languages)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        # Save
        output_file = self.output_dir / 'per_language_performance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {output_file}")
        
        plt.close()
    
    def generate_summary_report(self, comparison_df, results):
        """Generate text summary report."""
        print("Generating summary report...")
        
        report = []
        report.append("="*70)
        report.append("CROSS-LINGUAL RETRIEVAL SYSTEM - EVALUATION REPORT")
        report.append("="*70)
        
        report.append("="*70)
        report.append("OVERALL RESULTS")
        report.append("="*70)
        report.append(comparison_df.to_string(index=False, float_format='%.4f'))
        report.append("")
        
        # Findings: only state a comparison the data can support
        report.append("\n" + "="*70)
        report.append("KEY FINDINGS")
        report.append("="*70)
        statement, notes = superiority_conclusion(results)
        report.append(statement or "\nNo comparison can be drawn from these results.")
        for note in notes:
            report.append(f"  - {note}")
        
        # Per-language insights
        report.append("\n" + "="*70)
        report.append("PER-LANGUAGE PERFORMANCE")
        report.append("="*70)
        
        for method_name, data in results.items():
            if 'per_language_mrr' in data:
                report.append(f"\n{method_name.replace('_', ' ').title()}:")
                for lang, mrr in sorted(data['per_language_mrr'].items()):
                    report.append(f"  {lang.upper()}: {mrr:.4f}")
        
        report.append("\n" + "="*70 + "\n")
        
        # Save report
        report_text = '\n'.join(report)
        output_file = self.output_dir / 'evaluation_report.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Saved report to {output_file}")
        
        # Also print to console
        print("\n" + report_text)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare retrieval methods'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(results_dir=args.results_dir)
    
    # Load results
    results = evaluator.load_results()
    
    # Create comparison table
    comparison_df = evaluator.create_comparison_table(results)
    
    # Generate plots
    evaluator.plot_overall_comparison(comparison_df)
    evaluator.plot_per_language_performance(results)
    
    # Generate report
    evaluator.generate_summary_report(comparison_df, results)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved in: {args.results_dir}/evaluation/\n")


if __name__ == '__main__':
    main()
