import json
import logging
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import summarize_text, extract_incident_text, load_incidents
from datetime import datetime

# Load configuration from YAML file
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize logging
logging.basicConfig(level=logging.DEBUG if config["debug"] else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def summarize_incident(incident):
    guid = incident.get("guid")
    logging.info(f"Processing incident with GUID: {guid}")
    incident_text = extract_incident_text(incident)
    logging.debug(f"Prompt for T5 summarization: {incident_text}")
    start_time = time.time()
    t5_summary = summarize_text(incident_text, model_name="t5")
    duration = (time.time() - start_time) * 1000  # Duration in milliseconds
    timestamp = datetime.now().isoformat()
    logging.info(f"T5 summarization for GUID {guid} took {duration:.2f} milliseconds.")
    logging.debug(f"T5 summary: {t5_summary}")
    return {
        "guid": guid,
        "summary": t5_summary,
        "duration_ms": duration,
        "timestamp": timestamp
    }

def process_incidents(filepath: str, max_incidents: int = 5, workers: int = 20) -> list:
    """
    Process the incidents and generate T5 summaries.
    
    :param filepath: Path to the JSON file containing incident data.
    :param max_incidents: Maximum number of incidents to process for testing.
    :param workers: Number of concurrent workers to use.
    :return: A list of dictionaries with GUIDs, summaries, durations, and timestamps.
    """
    incidents = load_incidents(filepath, max_incidents)
    
    t5_summaries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(summarize_incident, incident) for incident in incidents]
        for future in as_completed(futures):
            try:
                t5_summaries.append(future.result())
            except Exception as e:
                logging.error(f"Error in future: {e}")
    
    logging.info("Generated T5 summaries for all incidents.")
    return t5_summaries

def save_t5_summaries(t5_summaries: list, output_filepath: str):
    """
    Save the T5 summaries to a JSON file.
    
    :param t5_summaries: A list of dictionaries with GUIDs, summaries, durations, and timestamps.
    :param output_filepath: Path to the output JSON file.
    """
    try:
        logging.info(f"Saving T5 summaries to {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(t5_summaries, file, indent=2)
        logging.info(f"Saved T5 summaries to {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving T5 summaries: {e}")

def main():
    """
    Main function to generate and save T5 summaries.
    """
    input_filepath = "data/data.json"
    output_filepath = "data/t5Summaries.json"
    max_incidents = int(sys.argv[1]) if len(sys.argv) > 1 else config.get("max_incidents", 5)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else config.get("concurrent_tasks", 20)
    
    logging.info("Starting T5 summarization process...")
    t5_summaries = process_incidents(input_filepath, max_incidents, workers)
    save_t5_summaries(t5_summaries, output_filepath)
    logging.info("T5 summarization process completed.")

if __name__ == "__main__":
    main()
