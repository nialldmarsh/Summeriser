import json
import logging
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from summarizer import Summarizer
from utils import extract_incident_text, load_incidents
import transformers

# Load configuration from YAML file
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize logging
logging.basicConfig(level=logging.DEBUG if config["debug"] else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific transformers warnings
transformers.logging.set_verbosity_error()

# Load BERT model
logging.info("Loading BERT model for summarization...")
model = Summarizer()
logging.info("BERT model loaded successfully.")

def summarize_incident(incident):
    guid = incident.get("guid")
    logging.info(f"Processing incident with GUID: {guid}")
    incident_text = extract_incident_text(incident)
    logging.debug(f"Prompt for BERT summarization: {incident_text}")
    start_time = time.time()
    bert_summary = model(incident_text, min_length=60, max_length=150)
    end_time = time.time()
    logging.info(f"BERT summarization for GUID {guid} took {end_time - start_time:.2f} seconds.")
    logging.debug(f"BERT summary: {bert_summary}")
    return guid, bert_summary

def process_incidents(filepath: str, max_incidents: int = 5, workers: int = 20) -> dict:
    """
    Process the incidents and generate BERT summaries.
    
    :param filepath: Path to the JSON file containing incident data.
    :param max_incidents: Maximum number of incidents to process for testing.
    :param workers: Number of concurrent workers to use.
    :return: A dictionary with GUIDs and their respective BERT summaries.
    """
    incidents = load_incidents(filepath, max_incidents)
    
    bert_summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(summarize_incident, incident) for incident in incidents]
        for future in as_completed(futures):
            try:
                guid, bert_summary = future.result()
                bert_summaries[guid] = bert_summary
            except Exception as e:
                logging.error(f"Error in future: {e}")
    
    logging.info("Generated BERT summaries for all incidents.")
    return bert_summaries

def save_bert_summaries(bert_summaries: dict, output_filepath: str):
    """
    Save the BERT summaries to a JSON file.
    
    :param bert_summaries: A dictionary with GUIDs and their respective BERT summaries.
    :param output_filepath: Path to the output JSON file.
    """
    try:
        logging.info(f"Saving BERT summaries to {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(bert_summaries, file, indent=2)
        logging.info(f"Saved BERT summaries to {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving BERT summaries: {e}")

def main():
    """
    Main function to generate and save BERT summaries.
    """
    input_filepath = "data/data.json"
    output_filepath = "data/bertSummaries.json"
    max_incidents = int(sys.argv[1]) if len(sys.argv) > 1 else config.get("max_incidents", 5)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else config.get("concurrent_tasks", 20)
    
    logging.info("Starting BERT summarization process...")
    bert_summaries = process_incidents(input_filepath, max_incidents, workers)
    save_bert_summaries(bert_summaries, output_filepath)
    logging.info("BERT summarization process completed.")

if __name__ == "__main__":
    main()
