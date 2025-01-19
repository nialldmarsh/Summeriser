import json
import logging
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import summarize_text, extract_incident_text, load_incidents

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
    end_time = time.time()
    logging.info(f"T5 summarization for GUID {guid} took {end_time - start_time:.2f} seconds.")
    logging.debug(f"T5 summary: {t5_summary}")
    return guid, t5_summary

def process_incidents(filepath: str, max_incidents: int = 5, workers: int = 20) -> dict:
    """
    Process the incidents and generate T5 summaries.
    
    :param filepath: Path to the JSON file containing incident data.
    :param max_incidents: Maximum number of incidents to process for testing.
    :param workers: Number of concurrent workers to use.
    :return: A dictionary with GUIDs and their respective T5 summaries.
    """
    incidents = load_incidents(filepath, max_incidents)
    
    t5_summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(summarize_incident, incident) for incident in incidents]
        for future in as_completed(futures):
            try:
                guid, t5_summary = future.result()
                t5_summaries[guid] = t5_summary
            except Exception as e:
                logging.error(f"Error in future: {e}")
    
    logging.info("Generated T5 summaries for all incidents.")
    return t5_summaries

def save_t5_summaries(t5_summaries: dict, output_filepath: str):
    """
    Save the T5 summaries to a JSON file.
    
    :param t5_summaries: A dictionary with GUIDs and their respective T5 summaries.
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
