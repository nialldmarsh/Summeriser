# Summarizer Metrics

This project computes various metrics to evaluate the quality and completeness of AI-generated summaries against incident data and human-written summaries.

## Data

Place your JSON data files in the `data` directory. The incident data file should be named `data.json` and the builder summaries file should be named `builderSummariesGPT4oMini.json`. The files should follow the structure provided in the examples below:

### Incident Data Example (`data/data.json`)

```json
[
    {
        "IncidentID": "INC-3001",
        "Who": "Alice Brown",
        "Department": "Finance",
        "guid": "7f1c3b47-8e1a-4c29-9b4f-c1e76c1d3e4a",
        "System": "Expense Claim Portal",
        "WhatWentWrong": "Receipts failed to upload, and the portal displayed a timeout error after the second attachment.",
        "WhatWereYouDoing": "Submitting multiple receipts from a client visit for reimbursement.",
        "How": "The system abruptly froze, showed a spinner for about a minute, then returned a 'Session Timed Out' message.",
        "Why": "A server-side upload limit was exceeded by larger-than-expected receipt file sizes.",
        "IdentifiedRisks": "- Delayed reimbursement cycle\n- Possible duplication of claims if employees retry submissions",
        "Mitigation": "Reactive: Instructed user to split receipts into smaller files.\nProactive: Increase server file size limits and add more detailed error feedback for large uploads.",
        "ResolutionDetails": "IT raised the maximum allowable attachment size on the portal. Alice was able to upload her receipts successfully without subsequent timeouts.",
        "Status": "Resolved",
        "ResolutionType": "First Line Support",
        "AdditionalNotes": "Subsequent tests showed no further upload issues for other Finance staff.",
        "HumanSummary": "This incident involved uploading multiple large receipt files into the Expense Claim Portal, causing repeated timeouts. The root cause was the server’s restrictive file size limit. After initial troubleshooting revealed that breaking the files into smaller chunks resolved the issue, IT raised the portal’s file size threshold to handle larger attachments seamlessly. This measure prevented further disruptions and streamlined the claims process for Finance.",
        "BuilderSummary": null
    }
    // ... more incidents ...
]
```

### Builder Summaries Example (`data/builderSummariesGPT4oMini.json`)

```json
[
    {
        "guid": "7f1c3b47-8e1a-4c29-9b4f-c1e76c1d3e4a",
        "builderSummary": "Incident ID: INC-3001\nReported by: Alice Brown from Finance\nSystem: Expense Claim Portal\nProblem: Receipts failed to upload, and the portal displayed a timeout error after the second attachment.\nAction Taken: The system abruptly froze, showed a spinner for about a minute, then returned a 'Session Timed Out' message.\nCause: A server-side upload limit was exceeded by larger-than-expected receipt file sizes.\nMitigation: Reactive: Instructed user to split receipts into smaller files.\nProactive: Increase server file size limits and add more detailed error feedback for large uploads.\nResolution: IT raised the maximum allowable attachment size on the portal. Alice was able to upload her receipts successfully without subsequent timeouts.\nStatus: Resolved\nAdditional Notes: Subsequent tests showed no further upload issues for other Finance staff."
    }
    // ... more summaries ...
]
```

### BERT Summaries Example (`data/bertSummaries.json`)

```json
[
    {
        "guid": "7f1c3b47-8e1a-4c29-9b4f-c1e76c1d3e4a",
        "bertSummary": "Incident ID: INC-3001\nReported by: Alice Brown from Finance\nSystem: Expense Claim Portal\nProblem: Receipts failed to upload, and the portal displayed a timeout error after the second attachment.\nAction Taken: The system abruptly froze, showed a spinner for about a minute, then returned a 'Session Timed Out' message.\nCause: A server-side upload limit was exceeded by larger-than-expected receipt file sizes.\nMitigation: Reactive: Instructed user to split receipts into smaller files.\nProactive: Increase server file size limits and add more detailed error feedback for large uploads.\nResolution: IT raised the maximum allowable attachment size on the portal. Alice was able to upload her receipts successfully without subsequent timeouts.\nStatus: Resolved\nAdditional Notes: Subsequent tests showed no further upload issues for other Finance staff."
    }
    // ... more summaries ...
]
```

## Metrics

### Content Coverage Metrics

1. **Completeness Score**
   - Calculation: `len(builder_summary) / len(incident_fields)`
   - Purpose: Measure how much of the incident information is captured in the builder summary
   - Range: 0.0 to ∞

2. **Precision, Recall, F1 Score**
   - Uses token-based comparison between builder summary and combined incident fields
   - Precision: `common_tokens / builder_tokens`
   - Recall: `common_tokens / incident_field_tokens`
   - F1: `2 * (precision * recall) / (precision + recall)`
   - Range: 0.0 to 1.0 (higher is better)

### Quality Comparison Metrics (Against Human Summary)

1. **Cosine Similarity** (Using scikit-learn)
   - Uses TF-IDF vectorization to compare semantic similarity
   - Calculation: Vector space similarity between summaries
   - Range: -1.0 to 1.0 (1.0 indicates identical content)

2. **BLEU Score** (Using NLTK)
   - Measures n-gram precision between builder and human summaries
   - Considers word order and phrase matching
   - Range: 0.0 to 1.0 (higher means better match)

### Human Comparison Metrics

1. **HumanCompletenessScore**
   - Calculation: `len(human_summary) / len(builder_summary)`
   - Purpose: Compare length ratios between human and builder summaries
   - Range: 0.0 to ∞

2. **HumanPrecision, HumanRecall**
   - Token-based comparison between human and builder summaries
   - Shows how well human summaries align with builder output
   - Range: 0.0 to 1.0

## Analysis Script

The `evaluate.py` script provides statistical analysis of the metrics:

### Content Coverage Analysis
- Averages completeness scores across all incidents
- Calculates mean F1, precision, and recall against source fields
- Helps evaluate if summaries consistently capture incident information

### Quality Score Analysis
- Averages BLEU and cosine similarity scores
- Indicates overall summary quality compared to human references

### Human Comparison Analysis
- Tracks average human-builder alignment metrics
- Shows typical length and content relationships
- Helps identify systematic differences between human and AI approaches

### Output Format
```
Summarization Metrics Analysis
=============================

Content Coverage:
Average Completeness Score: 0.XXX
Average F1 Score: 0.XXX
Average Precision: 0.XXX
Average Recall: 0.XXX

Quality Scores:
Average BLEU Score: 0.XXX
Average Cosine Similarity: 0.XXX

Human Comparison:
Average Human Completeness: 0.XXX
Average Human Precision: 0.XXX
Average Human Recall: 0.XXX

BERT Scores:
Average BERT Score: 0.XXX
```

## Running the Scripts

### Running Summarizers

To run the summarizers in parallel, use the `main.py` script. This script will run the T5, BART, and BERT summarizers concurrently and then evaluate the results.

```bash
python main.py
```

### Running Individual Summarizers

You can also run each summarizer individually if needed:

```bash
python t5_summarizer.py
python bart_summarizer.py
python bert_summarizer.py
```

### Evaluating Metrics

To compute metrics and analyze the results, use the `evaluate.py` script:

```bash
python evaluate.py
```

## License

This project is licensed under the MIT License.
