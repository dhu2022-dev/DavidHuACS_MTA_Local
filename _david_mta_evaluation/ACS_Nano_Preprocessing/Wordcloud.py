import json
import re
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load JSONL file and extract text data
def load_jsonl(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data.get('abstract', ''))  # Assuming the key we want is 'abstract'
    return texts

# Preprocess text using spaCy
def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Process text with spaCy
    doc = nlp(text)
    # Remove stopwords and punctuation, and apply lemmatization
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Preprocess a list of texts
def preprocess_texts(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    return ' '.join(processed_texts)

# Generate and display word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def main(json_file_path):
    texts = load_jsonl(json_file_path)
    processed_text = preprocess_texts(texts)
    generate_wordcloud(processed_text)

if __name__ == "__main__":
    # Replace 'your_file.jsonl' with the path to your JSONL file
    main('nano_title_abs_2yrs.jsonl')
