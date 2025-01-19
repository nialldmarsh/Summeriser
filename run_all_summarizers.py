import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from summarizers import T5Summarizer, BartSummarizer, BertSummarizer

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize summarizers
t5 = T5Summarizer()
bart = BartSummarizer()
bert = BertSummarizer()

def run_script(script_name: str, max_records: int = None):
    """
    Run a Python script with an optional maximum number of records.
    
    :param script_name: The name of the script to run.
    :param max_records: The maximum number of records to process.
    """
    try:
        logging.info(f"Running script: {script_name}")
        command = ["python", script_name]
        if max_records is not None:
            command.append(str(max_records))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Script {script_name} output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script {script_name}: {e.stderr}")

def summarize_incident(summarizer, incident):
    """
    Summarize a single incident using the specified summarizer.
    
    :param summarizer: The summarizer function to use (t5.summarize, bart.summarize, bert.summarize).
    :param incident: The incident data to summarize.
    :return: Summary and duration in milliseconds.
    """
    start_time = time.time()
    summary = summarizer(incident)
    duration = (time.time() - start_time) * 1000  # Duration in milliseconds
    return summary, duration

def summarize_all(incidents, config):
    """
    Summarize all incidents using the enabled summarizers in parallel.
    
    :param incidents: List of incident data.
    :param config: Configuration dictionary.
    :return: Dictionary of summaries with durations.
    """
    summaries = {
        't5': {},
        'bart': {},
        'bert': {}
    }
    
    summarizers = {
        't5': t5.summarize,
        'bart': bart.summarize,
        'bert': bert.summarize
    }
    
    max_workers = config.get('incident_workers', 5)  # Default to 5 workers if not specified
    
    for key, summarizer in summarizers.items():
        if config['summarizers'].get(key, False):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_guid = {executor.submit(summarize_incident, summarizer, incident): incident['guid'] for incident in incidents}
                for future in as_completed(future_to_guid):
                    guid = future_to_guid[future]
                    try:
                        summary, duration = future.result()
                        summaries[key][guid] = {
                            'summary': summary,
                            'duration_ms': duration
                        }
                    except Exception as e:
                        logging.error(f"Error summarizing incident {guid} with {key}: {e}")
                        summaries[key][guid] = {
                            'summary': "",
                            'duration_ms': 0
                        }
    return summaries

def summarize(text: str) -> dict:
    """
    Summarize the input text using all available summarizers.
    
    :param text: The text to be summarized.
    :return: A dictionary with summaries from each summarizer.
    """
    summaries = {}
    start_time = time.time()
    summaries['t5'] = t5.summarize(text)
    summaries['t5_duration'] = (time.time() - start_time) * 1000  # Duration in milliseconds
    
    start_time = time.time()
    summaries['bart'] = bart.summarize(text)
    summaries['bart_duration'] = (time.time() - start_time) * 1000  # Duration in milliseconds
    
    start_time = time.time()
    summaries['bert'] = bert.summarize(text)
    summaries['bert_duration'] = (time.time() - start_time) * 1000  # Duration in milliseconds
    
    return summaries

def main():
    """
    Main function to run all summarizer scripts in parallel.
    """
    max_records = int(sys.argv[1]) if len(sys.argv) > 1 else None
    scripts = ["t5_summarizer.py", "bart_summarizer.py", "bert_summarizer.py"]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_script, script, max_records) for script in scripts]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in future: {e}")

if __name__ == "__main__":
    # Example usage
    # Load incidents from a file or another source
    def load_incidents(file_path):
        """
        Load incidents from a JSON file.
        
        :param file_path: Path to the JSON file containing incidents.
        :return: List of incidents.
        """
        import json
        with open(file_path, 'r') as file:
            incidents = json.load(file)
        return incidents

    incidents = load_incidents("path_to_incidents.json")
    config = load_config("config.yml")
    summaries = summarize_all(incidents, config)
    save_summaries(summaries, "output/summaries.json")
    main()
