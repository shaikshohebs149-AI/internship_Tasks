import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text_ai(text):
    """
    Abstractive Summarization using Hugging Face Transformers (BART model).
    This generates a human-like summary by rephrasing.
    """
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# --- SAMPLE INPUT ---
article_text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction 
between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, 
understand, and make sense of the human languages in a manner that is valuable. Most NLP techniques rely on 
machine learning to derive meaning from human languages. Text summarization is one of the most exciting 
applications of NLP. It involves reducing a text document into a shorter version while preserving its 
information content and meaning. There are two main types: extractive and abstractive. Extractive 
summarization picks out the most important sentences, while abstractive summarization generates new 
sentences to convey the main idea.
"""

if __name__ == "__main__":
    print("--- ORIGINAL TEXT ---")
    print(article_text.strip())
    
    print("\n--- GENERATING SUMMARY... ---")
    result = summarize_text_ai(article_text)
    
    print("\n--- CONCISE SUMMARY ---")
    print(result)