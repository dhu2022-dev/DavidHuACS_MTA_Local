import json
import re
import unicodedata
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Load spaCy model
class SpaCyModelLoader:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_model(self):
        return self.nlp

# File Writing Transformer for debugging / visualization. Other transformers inherit it
class FileWritingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        self.filename = filename

    def fit(self, X, y=None):
        return self

    def write_to_file(self, data):
        with open(self.filename, 'w', encoding='utf-8') as f:
            for item in data:
                if isinstance(item, str):
                    f.write(item + '\n')
                elif hasattr(item, 'text'):
                    f.write(item.text + '\n')
                else:
                    f.write(str(item) + '\n')
    
    def transform(self, X, y=None):
        data = self._transform_data(X)
        self.write_to_file(data)
        print(f"{self.__class__.__name__} file written.")
        return data

    def _transform_data(self, X):
        raise NotImplementedError("Subclasses should implement this method")

# Clean Text Transformer
class CleanTextTransformer(FileWritingTransformer):
    def __init__(self, stop_words, filename='clean_text_output.txt'):
        super().__init__(filename)
        self.stop_words = stop_words

    def _transform_data(self, X):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
        # Ensure text is a string and encode/decode to handle any special characters
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        elif not isinstance(text, str):
            text = str(text, 'utf-8', errors='ignore')

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in self.stop_words)

        # Remove extra whitespaces
        text = text.strip()

        return text

# Tokenizer Transformer
class TokenizerTransformer(FileWritingTransformer):
    def __init__(self, nlp, filename='tokenized_output.txt'):
        super().__init__(filename)
        self.nlp = nlp

    def _transform_data(self, X):
        return [self.nlp(text) for text in X]

# Lemmatizer Transformer
class LemmatizerTransformer(FileWritingTransformer):
    def __init__(self, nlp, filename='lemmatized_output.txt'):
        super().__init__(filename)
        self.nlp = nlp

    def _transform_data(self, X):
        return [' '.join(token.lemma_ for token in doc) for doc in X]

# Normalizer Transformer
class NormalizeTextTransformer(FileWritingTransformer):
    def __init__(self, filename='normalized_output.txt'):
        super().__init__(filename)

    def _transform_data(self, X):
        return [unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8') for text in X]

# Topic Modeling Transformer
class TopicModelingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.lda = None
        self.dictionary = None

    def fit(self, X, y=None):
        # Ensure X is a list of tokenized documents (lists of words)
        texts = [text.split() for text in X]
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        self.lda = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=15)
        return self

    def transform(self, X):
        texts = [text.split() for text in X]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        topic_distribution = [self.lda.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        return topic_distribution
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# Load JSONL file and extract text data (combining title and abstract)
def load_jsonl(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            combined_text = f"{data.get('title', '')} {data.get('abstract', '')}"
            texts.append(combined_text)
    return texts

def main():
    # Final Pipeline for Text Preprocessing
    spacy_loader = SpaCyModelLoader()
    preprocessing_pipeline = Pipeline([
        ('clean', CleanTextTransformer(stop_words=set(spacy_loader.get_model().Defaults.stop_words))),
        ('tokenize', TokenizerTransformer(nlp=spacy_loader.get_model())),
        ('lemmatize', LemmatizerTransformer(nlp=spacy_loader.get_model())),
        ('normalize', NormalizeTextTransformer())
    ])

    # Load text data
    file_path = 'nano_title_abs_2yrs.jsonl'
    text_data = load_jsonl(file_path)

    # Apply the data to preprocessing pipeline
    preprocessed_data = preprocessing_pipeline.fit_transform(text_data)

    # Print the type and a preview of the preprocessed data
    print()
    print(f"Type of preprocessed_data: {type(preprocessed_data)}")
    if isinstance(preprocessed_data, list):
        print(f"Preview of preprocessed_data (first 5 items): {preprocessed_data[:5]}")
    elif hasattr(preprocessed_data, 'shape'):  # Check if it's a numpy array or similar
        print(f"Shape of preprocessed_data: {preprocessed_data.shape}")
        print(f"Preview of preprocessed_data (first 5 items): {preprocessed_data[:5]}")
    else:
        print(f"Preview of preprocessed_data: {preprocessed_data}")
    print()

    # Define the Topic Modeling Pipeline
    topic_modeling_pipeline = Pipeline([
        ('topic_model', TopicModelingTransformer())
    ])

    # Experiment with different numbers of topics
    coherence_scores = []
    for num_topics in range(2, 21):  # Arbitrary range, can adjust as needed
        print("Trying " + str(num_topics) + " topics.")

        # Update the number of topics
        topic_modeling_pipeline.named_steps['topic_model'].set_params(num_topics=num_topics)
        
        # Fit the topic modeling pipeline
        topic_modeling_pipeline.fit(preprocessed_data)
        
        # Extract the fitted topic model
        fitted_topic_model = topic_modeling_pipeline.named_steps['topic_model']
        
        # Calculate Coherence Score
        coherence_model = CoherenceModel(model=fitted_topic_model.lda, 
                                         texts=[text.split() for text in preprocessed_data], 
                                         dictionary=fitted_topic_model.dictionary, 
                                         coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append((num_topics, coherence_score))
        print(f'Coherence score: {coherence_score}')

    # 1. Find the best number of topics based on coherence scores
    best_num_topics, best_coherence = max(coherence_scores, key=lambda x: x[1])

    # 2. Refine and fit the Topic Modeling Transformer with the best number of topics
    final_topic_modeling_pipeline = Pipeline([
        ('topic_model', TopicModelingTransformer())
    ])
    final_topic_modeling_pipeline.named_steps['topic_model'].set_params(num_topics=best_num_topics)
    final_topic_modeling_pipeline.fit(preprocessed_data)

    # 3. Transform the data to get the topic distribution
    doc_topic_dist = final_topic_modeling_pipeline.transform(preprocessed_data)

    # 4. Convert the topic distribution to a dense matrix
    doc_topic_matrix = np.zeros((len(preprocessed_data), best_num_topics))
    for i, doc in enumerate(doc_topic_dist):
        for topic_num, prob in doc:
            doc_topic_matrix[i, topic_num] = prob

    # 5. Extract topics from the final model
    lda_model = final_topic_modeling_pipeline.named_steps['topic_model'].lda
    topics = lda_model.show_topics(num_topics=best_num_topics, formatted=False)

    # 6. Prepare texts for visualizations

    ### Visualization

    ### 1. Dimensionality Reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42)  # 2D visualization
    X_reduced = tsne.fit_transform(doc_topic_matrix)

    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of Document-Topic Distributions')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('tsne_visualization.png')
    plt.close()

    ### 2. Word Clouds for each topic
    for i, topic in enumerate(topics):
        words = dict(topic[1])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {i + 1}')
        plt.savefig(f'wordcloud_topic_{i + 1}.png')
        plt.close()

    ### 3. Calculate Term Frequencies
    term_freq = {}
    for text in preprocessed_data:
        for word in text.split():
            term_freq[word] = term_freq.get(word, 0) + 1

    # Sort by frequency and select the top 20 terms
    term_freq = dict(sorted(term_freq.items(), key=lambda item: item[1], reverse=True))
    df = pd.DataFrame(list(term_freq.items()), columns=['Term', 'Frequency'])
    df.set_index('Term', inplace=True)
    top_terms = df.head(20).transpose()

    # Plot Term Frequency Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_terms, annot=True, cmap='YlGnBu')
    plt.title('Heatmap of Most Frequent Terms')
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.savefig('term_frequency_heatmap.png')
    plt.close()

    ### 4. Compute Document Similarity Matrix
    similarity_matrix = cosine_similarity(doc_topic_matrix)
    similarity_df = pd.DataFrame(similarity_matrix)

    # Plot Document Similarity Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, cmap='viridis')
    plt.title(f'Document Similarity Matrix for {best_num_topics} Topics')
    plt.savefig('document_similarity_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()