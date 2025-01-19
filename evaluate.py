#!/usr/bin/env python3
"""
summarisation_metrics_analysis.py

An improved script for processing incidents, computing various summarisation metrics 
(including ROUGE), and generating analysis outputs in JSON format.

Key Features:
  - Argument Parsing for file paths and preprocessing toggles
  - BLEU, Cosine Similarity, Precision/Recall, F1, Completeness scores
  - ROUGE-1, ROUGE-2, ROUGE-L metrics
  - Uses a requirements.txt for easy environment setup
"""

import json
import logging
import argparse
from typing import Dict, List, Tuple

import numpy as np

# NLTK-based imports (for optional advanced tokenisation)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BLEU Score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ROUGE Score
from rouge_score import rouge_scorer

# BERTScore
from bert_score import score as bert_score

# Readability metrics
import textstat

# ========== LOGGING CONFIGURATION ========== #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Uncomment if you haven’t already downloaded NLTK data:
nltk.download('punkt')
nltk.download('stopwords')


# ========== TOKENISATION & NORMALISATION ========== #
def preprocess_text(text: str, do_lowercase: bool = True, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenise and optionally lowercase and remove stopwords from the input text.
    
    :param text: Original text to be tokenised.
    :param do_lowercase: Whether to convert text to lowercase.
    :param remove_stopwords: Whether to remove stopwords from the tokenised text.
    :return: A list of clean tokens.
    """
    if not text:
        return []

    if do_lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))  # Adjust language as needed
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


def text_pipeline(
    text: str,
    do_lowercase: bool = True,
    remove_stopwords: bool = False
) -> str:
    """
    Apply a unified text preprocessing approach, returning the cleaned text as a string.
    
    :param text: Input text.
    :param do_lowercase: Whether to convert text to lowercase.
    :param remove_stopwords: Whether to remove stopwords from the tokenised text.
    :return: Preprocessed text rejoined into a single string.
    """
    tokens = preprocess_text(text, do_lowercase=do_lowercase, remove_stopwords=remove_stopwords)
    return " ".join(tokens)


# ========== METRIC COMPUTATION FUNCTIONS ========== #
def compute_completeness_score(summary: str, reference: str) -> float:
    """
    Compute completeness score: len(summary) / len(reference).
    """
    if not summary or not reference:
        return 0.0
    reference_len = len(reference)
    return len(summary) / reference_len if reference_len > 0 else 0.0


def compute_cosine_similarity(summary: str, reference: str) -> float:
    """
    Compute cosine similarity between the summary and the reference using TF-IDF.
    """
    if not summary or not reference:
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([summary, reference]).toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1]


def compute_bleu_score(summary: str, reference: str) -> float:
    """
    Compute BLEU score between the summary and reference.
    """
    if not summary or not reference:
        return 0.0
    reference = [reference.split()]
    candidate = summary.split()
    try:
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
    except ZeroDivisionError:
        score = 0.0
    return score


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute F1 score from precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_precision_recall(summary: str, reference: str, do_preprocess: bool = False) -> Tuple[float, float]:
    """
    Compute token-based precision and recall between the summary and reference.
    """
    if not summary or not reference:
        return 0.0, 0.0

    if do_preprocess:
        summary_tokens = set(preprocess_text(summary))
        reference_tokens = set(preprocess_text(reference))
    else:
        summary_tokens = set(summary.split())
        reference_tokens = set(reference.split())

    common_tokens = summary_tokens.intersection(reference_tokens)
    precision = len(common_tokens) / len(summary_tokens) if summary_tokens else 0.0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0.0
    return precision, recall


def compute_rouge_scores(summary: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F-measures between summary and reference.

    :return: A dictionary with keys 'Rouge1', 'Rouge2', 'RougeL' and their respective F1 values.
    """
    if not summary or not reference:
        return {"Rouge1": 0.0, "Rouge2": 0.0, "RougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    # Each score contains precision, recall, and fmeasure. 
    return {
        "Rouge1": scores['rouge1'].fmeasure,
        "Rouge2": scores['rouge2'].fmeasure,
        "RougeL": scores['rougeL'].fmeasure
    }


def compute_bertscore(summary: str, reference: str):
    """
    Compute BERTScore between summary and reference.
    """
    if not summary or not reference:
        return 0.0
    P, R, F1 = bert_score([summary], [reference], lang="en", rescale_with_baseline=True)
    return F1.mean().item()


def compute_readability_metrics(text: str) -> Dict[str, float]:
    """
    Compute readability metrics for the given text.
    
    :param text: The text to be evaluated.
    :return: A dictionary with readability scores.
    """
    return {
        "FleschReadingEase": textstat.flesch_reading_ease(text),
        "FleschKincaidGrade": textstat.flesch_kincaid_grade(text),
        "GunningFogIndex": textstat.gunning_fog(text)
    }


# ========== LOADING AND SAVING DATA ========== #
def load_builder_summaries(filepath: str) -> dict:
    """
    Load builder summaries from a JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            summaries = json.load(file)
        logging.info(f"Loaded builder summaries from {filepath}")
        return {summary['guid']: summary['builderSummary'] for summary in summaries}
    except Exception as e:
        logging.error(f"Error loading builder summaries from {filepath}: {e}")
        return {}


def load_t5_summaries(filepath: str) -> dict:
    """
    Load T5 summaries from a JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            summaries = json.load(file)
        logging.info(f"Loaded T5 summaries from {filepath}")
        return summaries
    except Exception as e:
        logging.error(f"Error loading T5 summaries from {filepath}: {e}")
        return {}


def load_summaries(filepath: str) -> list:
    """
    Load summaries from a JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            summaries = json.load(file)
        logging.info(f"Loaded summaries from {filepath}")
        return summaries
    except Exception as e:
        logging.error(f"Error loading summaries from {filepath}: {e}")
        return []


def load_metrics(filepath: str) -> List[Dict]:
    """
    Load computed metrics from a JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Fixed unmatched parenthesis
            metrics = json.load(f)
        logging.info(f"Loaded metrics from {filepath}")
        return metrics
    except Exception as e:
        logging.error(f"Error loading metrics from {filepath}: {e}")
        return []


# ========== ANALYSIS & PRINTING FUNCTIONS ========== #
def analyze_metrics(metrics: List[Dict], evaluation_options: Dict) -> Dict:
    """
    Aggregate metrics across all incidents to produce summary statistics.
    """
    try:
        if not metrics:
            logging.warning("No metrics to analyze. Returning empty dictionary.")
            return {}

        analysis = {}

        logging.info("Metrics analysis completed")
        return analysis
    except Exception as e:
        logging.error(f"Error analyzing metrics: {e}")
        return {}


def print_analysis(analysis: Dict):
    """
    Print formatted analysis results to the console.
    """
    try:
        if not analysis:
            print("No analysis results to display.")
            return

        print("\nSummarization Metrics Analysis")
        print("=============================")

    except Exception as e:
        logging.error(f"Error printing analysis: {e}")


# ========== CORE PROCESSING FUNCTION ========== #
def compute_all_metrics_for_incident(
    builder_summary: str,
    human_summary: str,
    t5_summary: str,
    bart_summary: str,
    bert_summary: str,
    incident_fields: str,
    do_preprocess: bool,
    skip_bertscore: bool,
    durations: dict
) -> Dict[str, float]:
    """
    Compute all summarisation metrics (including ROUGE) for a single incident.
    This function unifies text preprocessing if required.
    """
    # If requested, apply a unified text pipeline to all text inputs
    if do_preprocess:
        builder = text_pipeline(builder_summary, do_lowercase=True, remove_stopwords=False)
        human = text_pipeline(human_summary, do_lowercase=True, remove_stopwords=False)
        t5 = text_pipeline(t5_summary, do_lowercase=True, remove_stopwords=False)
        bart = text_pipeline(bart_summary, do_lowercase=True, remove_stopwords=False)
        bert = text_pipeline(bert_summary, do_lowercase=True, remove_stopwords=False)
        incident = text_pipeline(incident_fields, do_lowercase=True, remove_stopwords=False)
    else:
        builder = builder_summary
        human = human_summary
        t5 = t5_summary
        bart = bart_summary
        bert = bert_summary
        incident = incident_fields

    # Metrics for builder summary
    builder_completeness_score = compute_completeness_score(builder, incident)
    builder_cosine_sim = compute_cosine_similarity(builder, human)
    builder_bleu_score = compute_bleu_score(builder, human)
    builder_precision, builder_recall = compute_precision_recall(builder, human, do_preprocess)
    builder_f1 = compute_f1_score(builder_precision, builder_recall)
    builder_rouge_dict = compute_rouge_scores(builder, human)
    builder_bertscore = compute_bertscore(builder, human) if not skip_bertscore else 0.0
    builder_readability = compute_readability_metrics(builder)

    # Metrics for T5 summary
    t5_completeness_score = compute_completeness_score(t5, incident)
    t5_cosine_sim = compute_cosine_similarity(t5, human)
    t5_bleu_score = compute_bleu_score(t5, human)
    t5_precision, t5_recall = compute_precision_recall(t5, human, do_preprocess)
    t5_f1 = compute_f1_score(t5_precision, t5_recall)
    t5_rouge_dict = compute_rouge_scores(t5, human)
    t5_bertscore = compute_bertscore(t5, human) if not skip_bertscore else 0.0
    t5_readability = compute_readability_metrics(t5)

    # Metrics for BART summary
    bart_completeness_score = compute_completeness_score(bart, incident)
    bart_cosine_sim = compute_cosine_similarity(bart, human)
    bart_bleu_score = compute_bleu_score(bart, human)
    bart_precision, bart_recall = compute_precision_recall(bart, human, do_preprocess)
    bart_f1 = compute_f1_score(bart_precision, bart_recall)
    bart_rouge_dict = compute_rouge_scores(bart, human)
    bart_bertscore = compute_bertscore(bart, human) if not skip_bertscore else 0.0
    bart_readability = compute_readability_metrics(bart)

    # Metrics for BERT summary
    bert_completeness_score = compute_completeness_score(bert, incident)
    bert_cosine_sim = compute_cosine_similarity(bert, human)
    bert_bleu_score = compute_bleu_score(bert, human)
    bert_precision, bert_recall = compute_precision_recall(bert, human, do_preprocess)
    bert_f1 = compute_f1_score(bert_precision, bert_recall)
    bert_rouge_dict = compute_rouge_scores(bert, human)
    bert_bertscore = compute_bertscore(bert, human) if not skip_bertscore else 0.0
    bert_readability = compute_readability_metrics(bert)

    # Metrics for human summary
    human_completeness_score = compute_completeness_score(human, incident)
    human_precision, human_recall = compute_precision_recall(human, incident, do_preprocess)
    human_f1 = compute_f1_score(human_precision, human_recall)
    human_readability = compute_readability_metrics(human)

    return {
        "BuilderCompletenessScore": builder_completeness_score,
        "BuilderCosineSimilarity": builder_cosine_sim,
        "BuilderBLEU": builder_bleu_score,
        "BuilderF1Score": builder_f1,
        "BuilderPrecision": builder_precision,
        "BuilderRecall": builder_recall,
        "BuilderRouge1": builder_rouge_dict["Rouge1"],
        "BuilderRouge2": builder_rouge_dict["Rouge2"],
        "BuilderRougeL": builder_rouge_dict["RougeL"],
        "BuilderBERTScore": builder_bertscore,
        "BuilderFleschReadingEase": builder_readability["FleschReadingEase"],
        "BuilderFleschKincaidGrade": builder_readability["FleschKincaidGrade"],
        "BuilderGunningFogIndex": builder_readability["GunningFogIndex"],
        "T5CompletenessScore": t5_completeness_score,
        "T5CosineSimilarity": t5_cosine_sim,
        "T5BLEU": t5_bleu_score,
        "T5F1Score": t5_f1,
        "T5Precision": t5_precision,
        "T5Recall": t5_recall,
        "T5Rouge1": t5_rouge_dict["Rouge1"],
        "T5Rouge2": t5_rouge_dict["Rouge2"],
        "T5RougeL": t5_rouge_dict["RougeL"],
        "T5BERTScore": t5_bertscore,
        "T5FleschReadingEase": t5_readability["FleschReadingEase"],
        "T5FleschKincaidGrade": t5_readability["FleschKincaidGrade"],
        "T5GunningFogIndex": t5_readability["GunningFogIndex"],
        "BARTCompletenessScore": bart_completeness_score,
        "BARTCosineSimilarity": bart_cosine_sim,
        "BARTBLEU": bart_bleu_score,
        "BARTF1Score": bart_f1,
        "BARTPrecision": bart_precision,
        "BARTRecall": bart_recall,
        "BARTRouge1": bart_rouge_dict["Rouge1"],
        "BARTRouge2": bart_rouge_dict["Rouge2"],
        "BARTRougeL": bart_rouge_dict["RougeL"],
        "BARTBERTScore": bart_bertscore,
        "BARTFleschReadingEase": bart_readability["FleschReadingEase"],
        "BARTFleschKincaidGrade": bart_readability["FleschKincaidGrade"],
        "BARTGunningFogIndex": bart_readability["GunningFogIndex"],
        "BERTCompletenessScore": bert_completeness_score,
        "BERTCosineSimilarity": bert_cosine_sim,
        "BERTBLEU": bert_bleu_score,
        "BERTF1Score": bert_f1,
        "BERTPrecision": bert_precision,
        "BERTRecall": bert_recall,
        "BERTRouge1": bert_rouge_dict["Rouge1"],
        "BERTRouge2": bert_rouge_dict["Rouge2"],
        "BERTRougeL": bert_rouge_dict["RougeL"],
        "BERTBERTScore": bert_bertscore,
        "BERTFleschReadingEase": bert_readability["FleschReadingEase"],
        "BERTFleschKincaidGrade": bert_readability["FleschKincaidGrade"],
        "BERTGunningFogIndex": bert_readability["GunningFogIndex"],
        "HumanCompletenessScore": human_completeness_score,
        "HumanPrecision": human_precision,
        "HumanRecall": human_recall,
        "HumanF1Score": human_f1,
        "HumanFleschReadingEase": human_readability["FleschReadingEase"],
        "HumanFleschKincaidGrade": human_readability["FleschKincaidGrade"],
        "HumanGunningFogIndex": human_readability["GunningFogIndex"],
        "T5Duration": durations.get('t5_duration', 0),
        "BARTDuration": durations.get('bart_duration', 0),
        "BERTDuration": durations.get('bert_duration', 0)
    }


def process_incidents(json_str: str, builder_summaries: dict, t5_summaries: list, bart_summaries: list, bert_summaries: list, do_preprocess: bool = False, skip_bertscore: bool = False, total_count: int = None) -> str:
    """
    Process the incidents, computing various metrics comparing builder summaries,
    T5 summaries, BART summaries, BERT summaries, and human summaries to the incident fields.
    
    :param json_str: String containing JSON data for incidents.
    :param builder_summaries: A dictionary mapping guids to builder summaries.
    :param t5_summaries: A list of dictionaries with T5 summaries.
    :param bart_summaries: A list of dictionaries with BART summaries.
    :param bert_summaries: A list of dictionaries with BERT summaries.
    :param do_preprocess: Whether to apply text preprocessing to all summaries and fields.
    :param skip_bertscore: Whether to skip BERTScore computation.
    :param total_count: Total number of incidents to process.
    :return: A JSON string of metrics for each incident.
    """
    try:
        incidents = json.loads(json_str)
        if not isinstance(incidents, list):
            logging.warning("JSON does not represent a list of incidents.")
            return "[]"
        results = []

        logging.debug(f"Available builder summary GUIDs: {list(builder_summaries.keys())}")
        logging.debug(f"Available T5 summary GUIDs: {[summary['guid'] for summary in t5_summaries]}")
        logging.debug(f"Available BART summary GUIDs: {[summary['guid'] for summary in bart_summaries]}")
        logging.debug(f"Available BERT summary GUIDs: {[summary['guid'] for summary in bert_summaries]}")

        for i, inc in enumerate(incidents):
            if total_count and i >= total_count:
                break
            guid = inc.get("guid")
            builder = builder_summaries.get(guid, "")
            human = inc.get("HumanSummary", "")
            t5_summary = next((summary for summary in t5_summaries if summary['guid'] == guid), {})
            bart_summary = next((summary for summary in bart_summaries if summary['guid'] == guid), {})
            bert_summary = next((summary for summary in bert_summaries if summary['guid'] == guid), {})

            # Concatenate relevant fields
            incident_fields = " ".join(filter(None, [
                inc.get("WhatWentWrong", ""),
                inc.get("WhatWereYouDoing", ""),
                inc.get("How", ""),
                inc.get("Why", ""),
                inc.get("IdentifiedRisks", ""),
                inc.get("Mitigation", ""),
                inc.get("ResolutionDetails", "")
            ])).strip()

            # Extract durations and timestamps
            durations = {
                't5_duration': t5_summary.get('duration_ms', 0),
                'bart_duration': bart_summary.get('duration_ms', 0),
                'bert_duration': bert_summary.get('duration_ms', 0)
            }
            timestamps = {
                't5_timestamp': t5_summary.get('timestamp', ''),
                'bart_timestamp': bart_summary.get('timestamp', ''),
                'bert_timestamp': bert_summary.get('timestamp', '')
            }

            # Compute all metrics
            metrics = compute_all_metrics_for_incident(builder, human, t5_summary.get('summary', ''), bart_summary.get('summary', ''), bert_summary.get('summary', ''), incident_fields, do_preprocess, skip_bertscore, durations)

            # Append results
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
                "BuilderSummary": builder,
                "HumanSummary": human,
                "T5Summary": t5_summary.get('summary', ''),
                "BARTSummary": bart_summary.get('summary', ''),
                "BERTSummary": bert_summary.get('summary', ''),
                **metrics,
                **timestamps
            })

        logging.info("Processed incidents and computed metrics.")
        return json.dumps(results, indent=2)
    except Exception as e:
        logging.error(f"Error processing incidents: {e}")
        return "[]"


# ========== MAIN FUNCTION ========== #
def main():
    """
    Main function demonstrating how to use the above utilities to process 
    summarisation metrics (including ROUGE) and analyse results.
    """
    parser = argparse.ArgumentParser(description="Summarisation Metrics Analysis with ROUGE")
    parser.add_argument("--incidents", type=str, default="data/data.json",
                        help="Path to the JSON file containing incident data.")
    parser.add_argument("--builder_summaries", type=str, default="data/builderSummariesGPT4oMini.json",
                        help="Path to the JSON file containing builder summaries.")
    parser.add_argument("--t5_summaries", type=str, default="data/t5Summaries.json",
                        help="Path to the JSON file containing T5 summaries.")
    parser.add_argument("--bart_summaries", type=str, default="data/bartSummaries.json",
                        help="Path to the JSON file containing BART summaries.")
    parser.add_argument("--bert_summaries", type=str, default="data/bertSummaries.json",
                        help="Path to the JSON file containing BERT summaries.")
    parser.add_argument("--output_metrics", type=str, default="output/metrics_output.json",
                        help="Path to output the computed metrics JSON.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply tokenisation and lowercasing before computing precision and recall.")
    parser.add.argument("--skip_bertscore", action="store_true",
                        help="Skip BERTScore computation.")
    parser.add.argument("--total_count", type=int, default=None,
                        help="Total number of incidents to process.")
    parser.add.argument("--evaluation_options", type=str, default="{}",
                        help="JSON string specifying which evaluation metrics to compute.")
    args = parser.parse_args()

    try:
        with open(args.incidents, "r", encoding='utf-8', errors='ignore') as file:
            json_data = file.read()
    except FileNotFoundError:
        logging.error(f"Incidents file not found: {args.incidents}")
        return

    builder_summaries = load_builder_summaries(args.builder_summaries)
    t5_summaries = load_t5_summaries(args.t5_summaries)
    bart_summaries = load_summaries(args.bart_summaries)
    bert_summaries = load_summaries(args.bert_summaries)

    # Process incidents and compute metrics
    metrics_json_str = process_incidents(json_data, builder_summaries, t5_summaries, bart_summaries, bert_summaries, do_preprocess=args.preprocess, skip_bertscore=args.skip_bertscore, total_count=args.total_count)

    # Save metrics
    try:
        with open(args.output_metrics, "w", encoding='utf-8') as output_file:
            output_file.write(metrics_json_str)
        logging.info(f"Saved metrics output to {args.output_metrics}")
    except Exception as e:
        logging.error(f"Error saving metrics output: {e}")
        return

    # Load metrics
    metrics = load_metrics(args.output_metrics)

    # Parse evaluation options
    try:
        evaluation_options = json.loads(args.evaluation_options)
    except json.JSONDecodeError:
        logging.error("Invalid JSON string for evaluation options. Using default options.")
        evaluation_options = {}

    # Analyse metrics (including ROUGE)
    analysis = analyze_metrics(metrics, evaluation_options)

    # Print analysis to console
    print_analysis(analysis)

    logging.info("Evaluation script started.")
    logging.info("Evaluation script completed.")

def evaluate_main(max_incidents: int = 100, evaluation_options: dict = {}):
    """
    Main function to evaluate the summarization metrics.
    
    :param max_incidents: Maximum number of incidents to process.
    :param evaluation_options: Dictionary specifying which evaluation metrics to compute.
    """
    parser = argparse.ArgumentParser(description="Summarisation Metrics Analysis with ROUGE")
    parser.add_argument("--incidents", type=str, default="data/data.json",
                        help="Path to the JSON file containing incident data.")
    parser.add_argument("--builder_summaries", type=str, default="data/builderSummariesGPT4oMini.json",
                        help="Path to the JSON file containing builder summaries.")
    parser.add_argument("--t5_summaries", type=str, default="data/t5Summaries.json",
                        help="Path to the JSON file containing T5 summaries.")
    parser.add_argument("--bart_summaries", type=str, default="data/bartSummaries.json",
                        help="Path to the JSON file containing BART summaries.")
    parser.add_argument("--bert_summaries", type=str, default="data/bertSummaries.json",
                        help="Path to the JSON file containing BERT summaries.")
    parser.add_argument("--output_metrics", type=str, default="output/metrics_output.json",
                        help="Path to output the computed metrics JSON.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply tokenisation and lowercasing before computing precision and recall.")
    parser.add_argument("--skip_bertscore", action="store_true",
                        help="Skip BERTScore computation.")
    parser.add_argument("--total_count", type=int, default=max_incidents,
                        help="Total number of incidents to process.")
    parser.add_argument("--evaluation_options", type=str, default=json.dumps(evaluation_options),
                        help="JSON string specifying which evaluation metrics to compute.")
    args = parser.parse_args()

    try:
        with open(args.incidents, "r", encoding='utf-8', errors='ignore') as file:
            json_data = file.read()
    except FileNotFoundError:
        logging.error(f"Incidents file not found: {args.incidents}")
        return

    builder_summaries = load_builder_summaries(args.builder_summaries)
    t5_summaries = load_t5_summaries(args.t5_summaries)
    bart_summaries = load_summaries(args.bart_summaries)
    bert_summaries = load_summaries(args.bert_summaries)

    # Process incidents and compute metrics
    metrics_json_str = process_incidents(json_data, builder_summaries, t5_summaries, bart_summaries, bert_summaries, do_preprocess=args.preprocess, skip_bertscore=args.skip_bertscore, total_count=args.total_count)

    # Save metrics
    try:
        with open(args.output_metrics, "w", encoding='utf-8') as output_file:
            output_file.write(metrics_json_str)
        logging.info(f"Saved metrics output to {args.output_metrics}")
    except Exception as e:
        logging.error(f"Error saving metrics output: {e}")
        return

    # Load metrics
    metrics = load_metrics(args.output_metrics)

    # Parse evaluation options
    try:
        evaluation_options = json.loads(args.evaluation_options)
    except json.JSONDecodeError:
        logging.error("Invalid JSON string for evaluation options. Using default options.")
        evaluation_options = {}

    # Analyse metrics (including ROUGE)
    analysis = analyze_metrics(metrics, evaluation_options)

    # Print analysis to console
    analysis = analyze_metrics(metrics, evaluation_options)
    print_analysis(analysis)

    # Log the analysis summary
    logging.info("Summarization Metrics Analysis Summary:")
    logging.info(json.dumps(analysis, indent=2))

    logging.info("Evaluation script started.")
    logging.info("Evaluation script completed.")

if __name__ == "__main__":
    main()

    # Print analysis to console
    print_analysis(analysis)

    # Log the analysis summary
    logging.info("Summarization Metrics Analysis Summary:")
    logging.info(json.dumps(analysis, indent=2))

    logging.info("Evaluation script started.")
    logging.info("Evaluation script completed.")

if __name__ == "__main__":
    main()