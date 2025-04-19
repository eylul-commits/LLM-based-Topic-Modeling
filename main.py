from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from gensim import corpora, models
from gensim.models import CoherenceModel


nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data[:1000]  # Use a subset to speed things up

# Preprocessing
def preprocess(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

cleaned_texts = [preprocess(doc) for doc in texts]


# Tokenize
tokenized_texts = [doc.split() for doc in cleaned_texts]

# Dictionary & Corpus
dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# Train LDA model
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

# Print topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"LDA Coherence Score: {coherence_lda:.4f}")


from bertopic import BERTopic

# You can skip preprocessing for BERTopic since it uses embeddings
topic_model = BERTopic(language="english", calculate_probabilities=True)
topics, _ = topic_model.fit_transform(texts)

# Show topics
print(topic_model.get_topic_info())
topic_model.visualize_topics().show()


# Get top words per topic
topics_words = topic_model.get_topics()

# Coherence via topic words (c_v using gensim)
bertopic_topics = []
for topic in topics_words.values():
    words = [word for word, _ in topic]
    bertopic_topics.append(words)

coherence_model_bertopic = CoherenceModel(topics=bertopic_topics, texts=tokenized_texts,
                                           dictionary=dictionary, coherence='c_v')
coherence_bertopic = coherence_model_bertopic.get_coherence()
print(f"BERTopic Coherence Score: {coherence_bertopic:.4f}")


from sklearn.metrics import silhouette_score
import numpy as np

embeddings = topic_model.embedding_model.embed(texts)
labels = topic_model.topics_

# Filter out noise points labeled as -1
valid_idx = [i for i, label in enumerate(labels) if label != -1]
sil_score = silhouette_score(np.array(embeddings)[valid_idx], [labels[i] for i in valid_idx])
print(f"Silhouette Score (BERTopic): {sil_score:.4f}")
