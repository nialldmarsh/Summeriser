import json
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

def extract_incident_text(incident: dict) -> str:
    """
    Extract relevant text from an incident for summarization.
    
    :param incident: A dictionary containing incident data.
    :return: A concatenated string of relevant incident fields.
    """
    return " ".join(filter(None, [
        f"Incident ID: {incident.get('IncidentID', '')}",
        f"Reported by: {incident.get('Who', '')} from {incident.get('Department', '')}",
        f"System: {incident.get('System', '')}",
        f"What Went Wrong: {incident.get('WhatWentWrong', '')}",
        f"What Were You Doing: {incident.get('WhatWereYouDoing', '')}",
        f"How: {incident.get('How', '')}",
        f"Why: {incident.get('Why', '')}",
        f"Identified Risks: {incident.get('IdentifiedRisks', '')}",
        f"Mitigation: {incident.get('Mitigation', '')}",
        f"Resolution Details: {incident.get('ResolutionDetails', '')}",
        f"Status: {incident.get('Status', '')}",
        f"Resolution Type: {incident.get('ResolutionType', '')}",
        f"Additional Notes: {incident.get('AdditionalNotes', '')}"
    ])).strip()

def load_incidents(filepath: str, max_incidents: int = 5) -> list:
    """
    Load incidents from a JSON file.
    
    :param filepath: Path to the JSON file containing incident data.
    :param max_incidents: Maximum number of incidents to load for testing.
    :return: A list of incidents.
    """
    try:
        logging.info(f"Loading incidents from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as file:
            incidents = json.load(file)
        return incidents[:max_incidents]
    except Exception as e:
        logging.error(f"Error loading incidents from {filepath}: {e}")
        return []

# Common detailed prompt for summarization
DETAILED_PROMPT = (
    "Generate a detailed summary of the following text. "
    "Include all key information, ensuring to mention full names and important details where applicable, "
    "and condense it into a coherent summary of 100-250 words: "
)

# Load T5 model and tokenizer
t5_model_name = "t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Load BART model and tokenizer
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

def summarize_text(text: str, model_name: str) -> str:
    """
    Summarize the input text using the specified model.
    
    :param text: The text to be summarized.
    :param model_name: The name of the model to use for summarization.
    :return: The summarized text.
    """
    if model_name == "t5":
        tokenizer = t5_tokenizer
        model = t5_model
    elif model_name == "bart":
        tokenizer = bart_tokenizer
        model = bart_model
    else:
        raise ValueError("Unsupported model name. Use 't5' or 'bart'.")

    prompt = DETAILED_PROMPT + text
    logging.debug(f"Using prompt: {prompt}")
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=250, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
