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
def load_config(config_path: str):
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

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
        "t5_summarizer.py",
        "bart_summarizer.py",
        "bert_summarizer.py",
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

if __name__ == "__main__":
    main()
