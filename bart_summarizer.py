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
    logging.debug(f"Prompt for BART summarization: {incident_text}")
    start_time = time.time()
    bart_summary = summarize_text(incident_text, model_name="bart")
    end_time = time.time()
    logging.info(f"BART summarization for GUID {guid} took {end_time - start_time:.2f} seconds.")
    logging.debug(f"BART summary: {bart_summary}")
    return guid, bart_summary

def process_incidents(filepath: str, max_incidents: int = 5, workers: int = 20) -> dict:
    """
    Process the incidents and generate BART summaries.
    
    :param filepath: Path to the JSON file containing incident data.
    :param max_incidents: Maximum number of incidents to process for testing.
    :param workers: Number of concurrent workers to use.
    :return: A dictionary with GUIDs and their respective BART summaries.
    """
    incidents = load_incidents(filepath, max_incidents)
    
    bart_summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(summarize_incident, incident) for incident in incidents]
        for future in as_completed(futures):
            try:
                guid, bart_summary = future.result()
                bart_summaries[guid] = bart_summary
            except Exception as e:
                logging.error(f"Error in future: {e}")
    
    logging.info("Generated BART summaries for all incidents.")
    return bart_summaries

def save_bart_summaries(bart_summaries: dict, output_filepath: str):
    """
    Save the BART summaries to a JSON file.
    
    :param bart_summaries: A dictionary with GUIDs and their respective BART summaries.
    :param output_filepath: Path to the output JSON file.
    """
    try:
        logging.info(f"Saving BART summaries to {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(bart_summaries, file, indent=2)
        logging.info(f"Saved BART summaries to {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving BART summaries: {e}")

def main():
    """
    Main function to generate and save BART summaries.
    """
    input_filepath = "data/data.json"
    output_filepath = "data/bartSummaries.json"
    max_incidents = int(sys.argv[1]) if len(sys.argv) > 1 else config.get("max_incidents", 5)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else config.get("concurrent_tasks", 20)
    
    logging.info("Starting BART summarization process...")
    bart_summaries = process_incidents(input_filepath, max_incidents, workers)
    save_bart_summaries(bart_summaries, output_filepath)
    logging.info("BART summarization process completed.")

if __name__ == "__main__":
    main()
