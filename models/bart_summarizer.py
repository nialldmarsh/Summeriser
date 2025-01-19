# BART summarizer implementation

from transformers import BartForConditionalGeneration, BartTokenizer

class BartSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
