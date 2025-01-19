import logging
import subprocess
import os
import yaml
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate import evaluate_main
from models.t5_summarizer import T5Summarizer
from models.bart_summarizer import BartSummarizer
from models.bert_summarizer import BertSummarizer

# Load configuration from YAML file
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize logging
logging.basicConfig(level=logging.DEBUG if config["debug"] else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_summarizer(script_name: str, max_incidents: int, workers: int):
    """
    Run a summarizer script and log progress.
    
    :param script_name: The name of the summarizer script to run.
    :param max_incidents: The maximum number of incidents to process.
    :param workers: The number of workers to use for this summarizer.
    """
    logging.info(f"Running {script_name} with max_incidents={max_incidents} and workers={workers}...")
    # Activate the virtual environment and run the script
    activate_venv = os.path.join(".venv", "Scripts", "activate.bat")
    command = f'cmd /c "{activate_venv} && python {script_name} {max_incidents} {workers}"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        logging.info(f"{script_name} ran successfully.")
    else:
        logging.error(f"Error running {script_name}: {result.stderr}")

def summarize(text):
    """
    Summarize the given text using all configured summarizers.
    
    :param text: The text to summarize.
    :return: A dictionary with summarizer names as keys and their summaries as values.
    """
    summaries = {}
    if config["summarizers"]["t5"]:
        t5_summarizer = T5Summarizer()
        summaries["t5"] = t5_summarizer.summarize(text)
    if config["summarizers"]["bart"]:
        bart_summarizer = BartSummarizer()
        summaries["bart"] = bart_summarizer.summarize(text)
    if config["summarizers"]["bert"]:
        bert_summarizer = BertSummarizer()
        summaries["bert"] = bert_summarizer.summarize(text)
    return summaries

def main():
    """
    Main function to run all summarizers and evaluate the results.
    """
    max_incidents = config.get("max_incidents", 100)
    total_workers = config.get("concurrent_tasks", 20)
    evaluation_options = config.get("evaluation_options", {})
    summarizer_scripts = []
    if config["summarizers"]["t5"]:
        summarizer_scripts.append("models/t5_summarizer.py")
    if config["summarizers"]["bart"]:
        summarizer_scripts.append("models/bart_summarizer.py")
    if config["summarizers"]["bert"]:
        summarizer_scripts.append("models/bert_summarizer.py")

    workers_per_summarizer = max(1, total_workers // len(summarizer_scripts)) if summarizer_scripts else 1

    # Use ThreadPoolExecutor to run summarizers in parallel
    with ThreadPoolExecutor(max_workers=len(summarizer_scripts)) as executor:
        futures = [executor.submit(run_summarizer, script, max_incidents, workers_per_summarizer) for script in summarizer_scripts]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in summarizer execution: {e}")

    # Summarization and duration logging
    sample_text = "Your incident report text goes here."
    results = summarize(sample_text)
    logging.info(f"Summarization Results: {results}")
    
    # Include durations in the results
    metrics_output = {
        "summaries": results,
        # ...other metrics...
    }
    
    # Write metrics to JSON
    os.makedirs("output", exist_ok=True)
    with open("output/metrics_output.json", "w") as outfile:
        json.dump(metrics_output, outfile, indent=4)

    if config["evaluate"]:
        logging.info("Running evaluation...")
        evaluate_main(max_incidents, evaluation_options)
        logging.info("Evaluation completed.")

if __name__ == "__main__":
    logging.info("Starting evaluation process...")
    main()
    logging.info("Evaluation process completed.")
