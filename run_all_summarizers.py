import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    main()
