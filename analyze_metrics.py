import json
import numpy as np
from typing import Dict, List

def load_metrics(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_metrics(metrics: List[Dict]) -> Dict:
    """Analyze metrics across all summaries to provide insights"""
    analysis = {
        'content_coverage': {
            'avg_completeness': np.mean([m['CompletenessScore'] for m in metrics]),
            'avg_f1': np.mean([m['F1Score'] for m in metrics]),
            'avg_precision': np.mean([m['Precision'] for m in metrics]),
            'avg_recall': np.mean([m['Recall'] for m in metrics])
        },
        'quality_scores': {
            'avg_bleu': np.mean([m['BLEU'] for m in metrics]),
            'avg_cosine': np.mean([m['CosineSimilarity'] for m in metrics])
        },
        'human_comparison': {
            'avg_human_completeness': np.mean([m['HumanCompletenessScore'] for m in metrics]),
            'avg_human_precision': np.mean([m['HumanPrecision'] for m in metrics]),
            'avg_human_recall': np.mean([m['HumanRecall'] for m in metrics])
        }
    }
    return analysis

def save_combined_output(metrics: List[Dict], analysis: Dict, filepath: str):
    """Save combined metrics and analysis results to a JSON file"""
    output = {
        "metrics": metrics,
        "analysis": analysis
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

def print_analysis(analysis: Dict):
    """Print formatted analysis results"""
    print("\nSummarization Metrics Analysis")
    print("=============================")
    
    print("\nContent Coverage:")
    print(f"Average Completeness Score: {analysis['content_coverage']['avg_completeness']:.3f}")
    print(f"Average F1 Score: {analysis['content_coverage']['avg_f1']:.3f}")
    print(f"Average Precision: {analysis['content_coverage']['avg_precision']:.3f}")
    print(f"Average Recall: {analysis['content_coverage']['avg_recall']:.3f}")
    
    print("\nQuality Scores:")
    print(f"Average BLEU Score: {analysis['quality_scores']['avg_bleu']:.3f}")
    print(f"Average Cosine Similarity: {analysis['quality_scores']['avg_cosine']:.3f}")
    
    print("\nHuman Comparison:")
    print(f"Average Human Completeness: {analysis['human_comparison']['avg_human_completeness']:.3f}")
    print(f"Average Human Precision: {analysis['human_comparison']['avg_human_precision']:.3f}")
    print(f"Average Human Recall: {analysis['human_comparison']['avg_human_recall']:.3f}")

if __name__ == "__main__":
    metrics = load_metrics("metrics_output.json")
    analysis = analyze_metrics(metrics)
    save_combined_output(metrics, analysis, "combined_output.json")
    print_analysis(analysis)
