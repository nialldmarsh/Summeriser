# Summarizer Metrics

## Introduction

Welcome to the Summarizer Metrics project! This repository is dedicated to evaluating AI-generated summaries against incident data and human-written summaries using various metrics. The goal is to ensure the quality, completeness, and reliability of automated summarization tools in processing and presenting critical information.

## Summarizer Modules

We have implemented three summarization models:

- **T5Summarizer**: Utilizes the T5 model for generating summaries.
- **BartSummarizer**: Employs the BART model for summarization tasks.
- **BertSummarizer**: Uses BERT for understanding and generating concise summaries.

## Model Summarization

We utilize advanced natural language processing models to generate concise and accurate summaries of incident reports. The primary models employed include:

- **T5**: A transformer-based model capable of generating fluent and contextually relevant summaries.
- **BART**: Combines bidirectional and autoregressive transformers for effective text generation.
- **BERT**: Utilized for embedding and semantic understanding to enhance summary relevance.

These models are fine-tuned on domain-specific data to optimize performance in summarizing structured incident reports.

## Tools and Technologies

The project leverages the following tools and frameworks:

- **Python**: The primary programming language used for scripting and automation.
- **Concurrent Futures**: Facilitates parallel execution of summarization tasks.
- **YAML**: For configuration management, allowing easy adjustments to parameters.
- **NLTK, scikit-learn, ROUGE, BERTScore, Textstat**: Libraries used for text processing and metric computations.
- **JSON**: Data interchange format for input and output data.
- **Logging**: For tracking the execution flow and debugging.

## Input Data

The system processes incident reports provided in JSON format. Each incident entry contains several properties, including but not limited to:

- **IncidentID**: Unique identifier for each incident.
- **Who**: Reporter of the incident.
- **Department**: Department responsible for the incident.
- **System**: System affected by the incident.
- **WhatWentWrong, WhatWereYouDoing, How, Why**: Detailed descriptions of the incident.
- **IdentifiedRisks, Mitigation, ResolutionDetails**: Risk factors and resolution strategies.
- **Status, ResolutionType, AdditionalNotes**: Additional metadata about the incident.

Example structure:
```json
[
    {
        "IncidentID": "INC-3001",
        "Who": "Alice Brown",
        "Department": "Finance",
        "guid": "7f1c3b47-8e1a-4c29-9b4f-c1e76c1d3e4a",
        "System": "Expense Claim Portal",
        "WhatWentWrong": "Receipts failed to upload...",
        // ...additional fields...
    },
    // ...more incidents...
]
```

## Configuration File

The `config.yml` file manages the project's settings, allowing customization of various parameters:

- **debug**: Enables or disables debug logging.
- **max_incidents**: Limits the number of incidents to process.
- **concurrent_tasks**: Sets the number of parallel summarization tasks.
- **summarizers**: Toggles the use of different summarization models (t5, bart, bert).
- **evaluate**: Determines whether to perform evaluation after summarization.
- **evaluation_options**: Specifies which metrics to compute (bleu, rouge, bertscore, readability, bert).
- **prompts**: Defines the prompt used for generating detailed summaries.
- **duration_tracking**: Enables tracking of summarizer execution durations.

Example:
```yaml
debug: false
max_incidents: 100
concurrent_tasks: 20
summarizers:
  t5: true
  bart: true
  bert: true
evaluate: true
evaluation_options:
  bleu: true
  rouge: true
  bertscore: true
  readability: true
  bert: true
prompts:
  detailed: "Provide a concise (up to 200 words)..."
duration_tracking: true
```

## Requirements

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

## Running the Scripts

To execute the summarization and evaluation process, follow these steps:

1. **Install Dependencies**:
   Ensure all required packages are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**:
   Adjust the `config.yml` file as needed to set parameters like the number of incidents, selected summarizers, evaluation metrics, and enable duration tracking.

3. **Run the Summarization Process**:
   Execute the main script to start summarizing incidents and evaluating the results:
   ```bash
   python main.py
   ```

4. **Output Metrics**:
   The metrics, including summarizer durations, will be saved in `output/metrics_output.json` and analyzed for insights.

## Output Metrics

The system generates a comprehensive set of metrics to evaluate the quality and completeness of the summaries:

### Content Coverage Metrics

1. **Completeness Score**
   - **Calculation**: `len(summary) / len(reference)`
   - **Purpose**: Measures how much of the incident information is captured in the summary.
   - **Range**: 0.0 to ∞

2. **Precision, Recall, F1 Score**
   - **Precision**: `common_tokens / summary_tokens`
   - **Recall**: `common_tokens / reference_tokens`
   - **F1 Score**: `2 * (precision * recall) / (precision + recall)`
   - **Purpose**: Evaluates the accuracy and coverage of the summary compared to the original incident.
   - **Range**: 0.0 to 1.0 (higher is better)

### Quality Comparison Metrics (Against Human Summary)

1. **Cosine Similarity**
   - **Calculation**: Using TF-IDF vectorization to compare semantic similarity.
   - **Range**: -1.0 to 1.0 (1.0 indicates identical content)

2. **BLEU Score**
   - **Calculation**: Measures n-gram precision between the summary and human reference.
   - **Range**: 0.0 to 1.0 (higher means better match)

### Human Comparison Metrics

1. **HumanCompletenessScore**
   - **Calculation**: `len(human_summary) / len(summary)`
   - **Purpose**: Compares the length ratios between human and AI-generated summaries.
   - **Range**: 0.0 to ∞

2. **HumanPrecision, HumanRecall**
   - **Calculation**: Token-based comparison between human and AI summaries.
   - **Range**: 0.0 to 1.0

### Readability Metrics

1. **Flesch Reading Ease**
2. **Flesch-Kincaid Grade**
3. **Gunning Fog Index**

   - **Purpose**: Assess the readability and complexity of the summaries.
   - **Range**: Higher scores generally indicate easier readability.

### Summarizer Duration Metrics

1. **T5 Duration**
   - **Purpose**: Time taken by the T5 summarizer to generate a summary.
   - **Range**: Measured in seconds.

2. **BART Duration**
   - **Purpose**: Time taken by the BART summarizer to generate a summary.
   - **Range**: Measured in seconds.

3. **BERT Duration**
   - **Purpose**: Time taken by the BERT summarizer to generate a summary.
   - **Range**: Measured in seconds.

## Understanding Scores

- **Good Scores**:
  - **Completeness Score**: Close to or exceeding 1.0, indicating thorough coverage of incident details.
  - **Precision, Recall, F1**: Values above 0.7 are considered strong, demonstrating accurate and comprehensive summaries.
  - **Cosine Similarity, BLEU Score**: Scores above 0.6 suggest high similarity to human-written summaries.
  - **Readability Metrics**: Balanced scores indicating clarity without oversimplification.
  - **Duration Metrics**: Lower durations indicate more efficient summarization.

- **Bad Scores**:
  - **Completeness Score**: Below 0.5, showing insufficient coverage of incident information.
  - **Precision, Recall, F1**: Values below 0.5 indicate poor summary quality.
  - **Cosine Similarity, BLEU Score**: Scores below 0.3 reflect low alignment with human references.
  - **Readability Metrics**: Extremely high or low scores may suggest either oversimplification or excessive complexity.
  - **Duration Metrics**: Higher durations may indicate inefficiency in summarization processes.

Regularly monitoring these metrics helps in refining the summarization models and ensuring the generated summaries meet the desired quality and performance standards.
