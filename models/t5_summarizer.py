from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Summarizer:
    def __init__(self, model_name='t5-small'):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
