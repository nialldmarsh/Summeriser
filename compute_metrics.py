import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from typing import Dict, List
from bert_score import score as bert_score

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to compute completeness score of builder summary relative to incident fields
def compute_completeness_score(builder_summary: str, incident_fields: str) -> float:
    if not builder_summary or not incident_fields:
        return 0.0
    return len(builder_summary) / len(incident_fields) if len(incident_fields) > 0 else 0.0

# Function to compute completeness score of human summary relative to builder summary
def compute_human_completeness_score(human_summary: str, builder_summary: str) -> float:
    if not human_summary or not builder_summary:
        return 0.0
    return len(human_summary) / len(builder_summary) if len(builder_summary) > 0 else 0.0

# Function to compute cosine similarity between builder and human summaries
def compute_cosine_similarity(builder_summary: str, human_summary: str):
    if not builder_summary or not human_summary:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform([builder_summary, human_summary])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

# Function to compute BLEU score between builder and human summaries
def compute_bleu_score(builder_summary: str, human_summary: str):
    if not builder_summary or not human_summary:
        return 0.0
    reference = [human_summary.split()]
    candidate = builder_summary.split()
    try:
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    except ZeroDivisionError:
        score = 0.0
    return score

# Function to compute F1 score using precision and recall
def compute_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Function to compute precision and recall between builder summary and incident fields
def compute_precision_recall(builder_summary: str, incident_fields: str):
    if not builder_summary or not incident_fields:
        return 0.0, 0.0
    builder_tokens = set(builder_summary.split())
    incident_tokens = set(incident_fields.split())
    common_tokens = builder_tokens.intersection(incident_tokens)
    precision = len(common_tokens) / len(builder_tokens) if builder_tokens else 0.0
    recall = len(common_tokens) / len(incident_tokens) if incident_tokens else 0.0
    return precision, recall

# Function to compute precision and recall for human summary relative to builder summary
def compute_human_precision_recall(human_summary: str, builder_summary: str):
    if not human_summary or not builder_summary:
        return 0.0, 0.0
    human_tokens = set(human_summary.split())
    builder_tokens = set(builder_summary.split())
    common_tokens = human_tokens.intersection(builder_tokens)
    precision = len(common_tokens) / len(human_tokens) if human_tokens else 0.0
    recall = len(common_tokens) / len(builder_tokens) if builder_tokens else 0.0
    return precision, recall

# Function to compute BERTScore between builder and human summaries
def compute_bertscore(builder_summary: str, human_summary: str):
    if not builder_summary or not human_summary:
        return 0.0
    P, R, F1 = bert_score([builder_summary], [human_summary], lang="en", rescale_with_baseline=True)
    return F1.mean().item()

# Function to load builder summaries from a JSON file
def load_builder_summaries(filepath: str) -> dict:
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            summaries = json.load(file)
        logging.info(f"Loaded builder summaries from {filepath}")
        return {summary['guid']: summary['builderSummary'] for summary in summaries}
    except Exception as e:
        logging.error(f"Error loading builder summaries from {filepath}: {e}")
        return {}

# Function to load metrics from a JSON file
def load_metrics(filepath: str) -> List[Dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        logging.info(f"Loaded metrics from {filepath}")
        return metrics
    except Exception as e:
        logging.error(f"Error loading metrics from {filepath}: {e}")
        return []

# Function to analyze metrics across all summaries to provide insights
def analyze_metrics(metrics: List[Dict]) -> Dict:
    try:
        analysis = {
            'content_coverage': {
                'avg_completeness': np.mean([m['CompletenessScore'] for m in metrics]),
                'avg_f1': np.mean([m['F1Score'] for m in metrics]),
                'avg_precision': np.mean([m['Precision'] for m in metrics]),
                'avg_recall': np.mean([m['Recall'] for m in metrics])
            },
            'quality_scores': {
                'avg_bleu': np.mean([m['BLEU'] for m in metrics]),
                'avg_cosine': np.mean([m['CosineSimilarity'] for m in metrics]),
                'avg_bertscore': np.mean([m['BERTScore'] for m in metrics])
            },
            'human_comparison': {
                'avg_human_completeness': np.mean([m['HumanCompletenessScore'] for m in metrics]),
                'avg_human_precision': np.mean([m['HumanPrecision'] for m in metrics]),
                'avg_human_recall': np.mean([m['HumanRecall'] for m in metrics])
            }
        }
        logging.info("Metrics analysis completed")
        return analysis
    except Exception as e:
        logging.error(f"Error analyzing metrics: {e}")
        return {}

# Function to save combined metrics and analysis results to a JSON file
def save_combined_output(metrics: List[Dict], analysis: Dict, filepath: str):
    try:
        output = {
            "metrics": metrics,
            "analysis": analysis
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logging.info(f"Saved combined output to {filepath}")
    except Exception as e:
        logging.error(f"Error saving combined output to {filepath}: {e}")

# Function to print formatted analysis results
def print_analysis(analysis: Dict):
    try:
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
        print(f"Average BERTScore: {analysis['quality_scores']['avg_bertscore']:.3f}")
        
        print("\nHuman Comparison:")
        print(f"Average Human Completeness: {analysis['human_comparison']['avg_human_completeness']:.3f}")
        print(f"Average Human Precision: {analysis['human_comparison']['avg_human_precision']:.3f}")
        print(f"Average Human Recall: {analysis['human_comparison']['avg_human_recall']:.3f}")
    except Exception as e:
        logging.error(f"Error printing analysis: {e}")

# Function to process incidents and compute various metrics
def process_incidents(json_str: str, builder_summaries: dict):
    try:
        incidents = json.loads(json_str)
        results = []

        logging.debug(f"Available builder summary GUIDs: {list(builder_summaries.keys())}")

        for inc in incidents:
            guid = inc.get("guid")
            builder = builder_summaries.get(guid, "")
            human = inc.get("HumanSummary", "")
            
            # Combine all relevant incident fields for completeness comparison
            incident_fields = (
                (inc.get("WhatWentWrong") or "") + " " +
                (inc.get("WhatWereYouDoing") or "") + " " +
                (inc.get("How") or "") + " " +
                (inc.get("Why") or "") + " " +
                (inc.get("IdentifiedRisks") or "") + " " +
                (inc.get("Mitigation") or "") + " " +
                (inc.get("ResolutionDetails") or "")
            ).strip()
            
            if not builder:
                logging.debug(f"No builder summary found for GUID {guid}")
            if not incident_fields:
                logging.debug(f"No incident fields found for GUID {guid}")
            
            logging.debug(f"Processing incident {guid}")
            logging.debug(f"Builder summary: {builder}")
            logging.debug(f"Human summary: {human}")
            
            # Compute various metrics
            completeness_score = compute_completeness_score(builder, incident_fields)
            cosine_sim = compute_cosine_similarity(builder, human)
            bleu_score = compute_bleu_score(builder, human)
            precision, recall = compute_precision_recall(builder, incident_fields)
            f1 = compute_f1_score(precision, recall)
            bertscore = compute_bertscore(builder, human)
            
            human_completeness_score = compute_human_completeness_score(human, builder)
            human_precision, human_recall = compute_human_precision_recall(human, builder)
            
            # Append results for each incident
            results.append({
                "GUID": guid,
                "IncidentID": inc.get("IncidentID"),
                "Who": inc.get("Who"),
                "Department": inc.get("Department"),
                "DepartmentId": inc.get("DepartmentId"),
                "System": inc.get("System"),
                "WhatWentWrong": inc.get("WhatWentWrong"),
                "WhatWereYouDoing": inc.get("WhatWereYouDoing"),
                "How": inc.get("How"),
                "Why": inc.get("Why"),
                "IdentifiedRisks": inc.get("IdentifiedRisks"),
                "Mitigation": inc.get("Mitigation"),
                "ResolutionDetails": inc.get("ResolutionDetails"),
                "Status": inc.get("Status"),
                "ResolutionType": inc.get("ResolutionType"),
                "AdditionalNotes": inc.get("AdditionalNotes"),
                "CompletenessScore": completeness_score,
                "CosineSimilarity": cosine_sim,
                "BLEU": bleu_score,
                "F1Score": f1,
                "Precision": precision,
                "Recall": recall,
                "BERTScore": bertscore,
                "HumanCompletenessScore": human_completeness_score,
                "HumanPrecision": human_precision,
                "HumanRecall": human_recall,
                "BuilderSummary": builder,
                "HumanSummary": human
            })
        
        logging.info("Processed incidents and computed metrics")
        return json.dumps(results, indent=2)
    except Exception as e:
        logging.error(f"Error processing incidents: {e}")
        return "[]"

def main():
    try:
        # Example usage with JSON files
        with open("data/data.json", "r", encoding='utf-8', errors='ignore') as file:
            json_data = file.read()
        
        builder_summaries = load_builder_summaries("data/builderSummariesGPT4oMini.json")
        metrics_json = process_incidents(json_data, builder_summaries)
        
        with open("output/metrics_output.json", "w", encoding='utf-8') as output_file:
            output_file.write(metrics_json)
        
        metrics = load_metrics("output/metrics_output.json")
        analysis = analyze_metrics(metrics)
        save_combined_output(metrics, analysis, "output/combined_output.json")
        print_analysis(analysis)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
