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
from run_all_summarizers import summarize_all, load_config, save_summaries

# Initialise logging
def initialise_logging(debug_mode: bool):
    """
    Set up logging configuration.

    :param debug_mode: Enable debug level logging if True.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Function to activate virtual environment and run scripts
def run_summarizer(script_name: str, max_incidents: int, workers: int):
    """
    Run a summarizer script and log progress.

    :param script_name: The name of the summarizer script to run.
    :param max_incidents: The maximum number of incidents to process.
    :param workers: The number of workers to use for this summarizer.
    """
    logging.info(f"Running {script_name} with max_incidents={max_incidents} and workers={workers}...")

    # Determine platform-specific virtual environment activation
    activate_venv = (
        os.path.join(".venv", "Scripts", "activate.bat")
        if os.name == "nt" else "source .venv/bin/activate"
    )

    command = f'{activate_venv} && python {script_name} {max_incidents} {workers}'

    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            logging.info(f"{script_name} ran successfully.")
        else:
            logging.error(f"Error running {script_name}: {result.stderr.strip()}")
    except Exception as e:
        logging.error(f"Failed to execute {script_name}: {e}")

# Main workflow
def main():
    """
    Main entry point for the summarisation workflow.
    """
    config = load_config("config.yml")
    initialise_logging(config.get("debug", False))

    max_incidents = config.get("max_incidents", 100)
    workers = config.get("workers", 4)

    summarizer_scripts = [
        "models/t5_summarizer.py",
        "models/bart_summarizer.py",
        "models/bert_summarizer.py",
    ]

    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(summarizer_scripts)) as executor:
        future_to_script = {
            executor.submit(run_summarizer, script, max_incidents, workers): script
            for script in summarizer_scripts
        }

        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            try:
                future.result()
                logging.info(f"Completed: {script_name}")
            except Exception as exc:
                logging.error(f"{script_name} generated an exception: {exc}")

    # Example usage
    def load_incidents(file_path):
        """
        Load incidents from a JSON file.

        :param file_path: Path to the JSON file containing incidents.
        :return: List of incidents.
        """
        try:
            with open(file_path, 'r') as file:
                incidents = json.load(file)
            return incidents
        except Exception as e:
            logging.error(f"Error loading incidents from {file_path}: {e}")
            return []

    incidents = load_incidents("path_to_incidents.json")
    summaries = summarize_all(incidents, config)
    save_summaries(summaries, "output/summaries.json")

    # Call the evaluation function
    evaluate_main(max_incidents=max_incidents, evaluation_options=config.get("evaluation_options", {}))

if __name__ == "__main__":
    main()
